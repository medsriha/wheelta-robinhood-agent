-- 0001_initial: ledger schema (ADR-0002, ADR-0010; docs/INTERFACES.md "Ledger tables and recovery").
--
-- Identity rows are inserted once. Observations and corrections are new event/evidence rows
-- (corrects_*_id); nothing is updated or deleted. Append-only is enforced by triggers at the
-- end of this file. Current state (run status, order status, workspace object state) is a
-- projection derived from events, never an updated column.
--
-- Closed sets mirror src/wheelta_robinhood_agent/domain/enums.py; event_type sets mirror
-- domain/events.py exactly (tests/unit/test_event_vocabulary.py). Changing an enum there
-- requires a new migration that replaces the corresponding CHECK constraint.
-- IDs are issued by code (UUIDv7) except runs.run_id (UUIDv5 of environment + slot).

-- ---------------------------------------------------------------------------------------------
-- Identity: runs
-- ---------------------------------------------------------------------------------------------
CREATE TABLE runs (
    run_id       uuid        PRIMARY KEY,
    environment  text        NOT NULL CHECK (environment IN ('local', 'staging', 'production')),
    slot         timestamptz NOT NULL
                 CHECK (date_trunc('hour', slot AT TIME ZONE 'UTC') = slot AT TIME ZONE 'UTC'),
    recorded_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT runs_environment_slot_key UNIQUE (environment, slot)
);

-- Run observations: start, recovery, status transitions, control latch, source status,
-- market session, prompt/model metadata, audit status. `status` is set only on status events.
CREATE TABLE run_events (
    event_id             uuid        PRIMARY KEY,
    entity_id            uuid        NOT NULL REFERENCES runs (run_id),
    run_id               uuid        NOT NULL REFERENCES runs (run_id),
    sequence             integer     NOT NULL CHECK (sequence > 0),
    event_type           text        NOT NULL CHECK (event_type IN (
                             'started', 'recovery_started', 'status', 'control',
                             'source_status', 'market_session', 'metadata', 'audit_status')),
    status               text        CHECK (status IN (
                             'running', 'completed', 'skipped_concurrent', 'skipped_killed',
                             'skipped_market_closed', 'stopped', 'timed_out', 'failed')),
    observed_at          timestamptz NOT NULL,
    source_as_of         timestamptz,
    source_tool_call_ids uuid[]      NOT NULL DEFAULT '{}',
    dedup_key            text        NOT NULL CHECK (length(dedup_key) > 0),
    payload              jsonb       NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(payload) = 'object'),
    corrects_event_id    uuid,
    recorded_at          timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT run_events_entity_is_run CHECK (entity_id = run_id),
    CONSTRAINT run_events_status_iff_status_event CHECK ((event_type = 'status') = (status IS NOT NULL)),
    CONSTRAINT run_events_entity_sequence_key UNIQUE (entity_id, sequence),
    CONSTRAINT run_events_entity_dedup_key UNIQUE (entity_id, dedup_key),
    CONSTRAINT run_events_event_entity_key UNIQUE (event_id, entity_id),
    CONSTRAINT run_events_corrects_same_entity
        FOREIGN KEY (corrects_event_id, entity_id) REFERENCES run_events (event_id, entity_id)
);

-- ---------------------------------------------------------------------------------------------
-- Identity: tool calls, and their evidence payloads
-- ---------------------------------------------------------------------------------------------
CREATE TABLE tool_calls (
    tool_call_id       uuid        PRIMARY KEY,
    run_id             uuid        NOT NULL REFERENCES runs (run_id),
    sdk_tool_use_id    text        NOT NULL CHECK (length(sdk_tool_use_id) > 0),
    stage              text        NOT NULL CHECK (length(stage) > 0),
    server             text        NOT NULL CHECK (length(server) > 0),
    tool               text        NOT NULL CHECK (length(tool) > 0),
    -- NULL: the tool is not in the registry (it has no tier and is denied).
    tier               text        CHECK (tier IN ('R', 'S', 'X', 'excluded')),
    requested_at       timestamptz NOT NULL,
    arguments_redacted jsonb       NOT NULL CHECK (jsonb_typeof(arguments_redacted) = 'object'),
    recorded_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT tool_calls_run_sdk_tool_use_key UNIQUE (run_id, sdk_tool_use_id)
);

-- Validated results, restricted redacted error payloads, raw invalid payloads, and the exact
-- envelope delivered to the model (INTERFACES.md ToolCallRecord result/error/delivered refs).
CREATE TABLE results (
    result_id          uuid        PRIMARY KEY,
    run_id             uuid        NOT NULL REFERENCES runs (run_id),
    tool_call_id       uuid        REFERENCES tool_calls (tool_call_id),
    kind               text        NOT NULL CHECK (kind IN (
                           'validated', 'error', 'raw_invalid', 'delivered')),
    payload            jsonb       NOT NULL,
    corrects_result_id uuid        REFERENCES results (result_id),
    recorded_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE tool_call_events (
    event_id                     uuid        PRIMARY KEY,
    entity_id                    uuid        NOT NULL REFERENCES tool_calls (tool_call_id),
    run_id                       uuid        NOT NULL REFERENCES runs (run_id),
    sequence                     integer     NOT NULL CHECK (sequence > 0),
    -- requested is committed before dispatch; dispatched records the allowed (possibly
    -- rewritten) arguments; the ToolCallStatus outcomes follow; delivered references the
    -- exact envelope handed to the model.
    event_type                   text        NOT NULL CHECK (event_type IN (
                                     'requested', 'dispatched', 'denied', 'succeeded',
                                     'failed', 'timed_out', 'unknown', 'delivered')),
    reason                       text,
    effective_arguments_redacted jsonb       CHECK (jsonb_typeof(effective_arguments_redacted) = 'object'),
    latency_ms                   integer     CHECK (latency_ms >= 0),
    request_id                   text,
    result_ref                   uuid        REFERENCES results (result_id),
    error_ref                    uuid        REFERENCES results (result_id),
    delivered_result_ref         uuid        REFERENCES results (result_id),
    observed_at                  timestamptz NOT NULL,
    source_as_of                 timestamptz,
    source_tool_call_ids         uuid[]      NOT NULL DEFAULT '{}',
    dedup_key                    text        NOT NULL CHECK (length(dedup_key) > 0),
    payload                      jsonb       NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(payload) = 'object'),
    corrects_event_id            uuid,
    recorded_at                  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT tool_call_events_denied_has_reason CHECK (event_type <> 'denied' OR reason IS NOT NULL),
    CONSTRAINT tool_call_events_entity_sequence_key UNIQUE (entity_id, sequence),
    CONSTRAINT tool_call_events_entity_dedup_key UNIQUE (entity_id, dedup_key),
    CONSTRAINT tool_call_events_event_entity_key UNIQUE (event_id, entity_id),
    CONSTRAINT tool_call_events_corrects_same_entity
        FOREIGN KEY (corrects_event_id, entity_id) REFERENCES tool_call_events (event_id, entity_id)
);

-- ---------------------------------------------------------------------------------------------
-- Identity: order intents, orders, positions, workspace objects
-- ---------------------------------------------------------------------------------------------
-- A placement intent has its own ID even when the broker order ID is unknown.
CREATE TABLE order_intents (
    intent_id          uuid        PRIMARY KEY,
    run_id             uuid        NOT NULL REFERENCES runs (run_id),
    place_tool_call_id uuid        NOT NULL REFERENCES tool_calls (tool_call_id),
    account_scope_id   text        NOT NULL CHECK (length(account_scope_id) > 0),
    arguments_redacted jsonb       NOT NULL CHECK (jsonb_typeof(arguments_redacted) = 'object'),
    recorded_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT order_intents_place_tool_call_key UNIQUE (place_tool_call_id)
);

-- Facts about a placement intent itself, needed when no broker order is (yet) linked:
-- `reviewed` links the review call that preceded the place call; `resolved_no_order` records
-- the broker reads proving no order exists for the intent (it then leaves owned_orders);
-- `reconciliation` is a note, or with corrects_event_id supersedes an earlier event.
CREATE TABLE order_intent_events (
    event_id             uuid        PRIMARY KEY,
    entity_id            uuid        NOT NULL REFERENCES order_intents (intent_id),
    run_id               uuid        NOT NULL REFERENCES runs (run_id),
    sequence             integer     NOT NULL CHECK (sequence > 0),
    event_type           text        NOT NULL CHECK (event_type IN (
                             'reviewed', 'resolved_no_order', 'reconciliation')),
    observed_at          timestamptz NOT NULL,
    source_as_of         timestamptz,
    source_tool_call_ids uuid[]      NOT NULL DEFAULT '{}',
    dedup_key            text        NOT NULL CHECK (length(dedup_key) > 0),
    payload              jsonb       NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(payload) = 'object'),
    corrects_event_id    uuid,
    recorded_at          timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT order_intent_events_evidence_has_sources CHECK (
        event_type NOT IN ('reviewed', 'resolved_no_order') OR cardinality(source_tool_call_ids) > 0),
    CONSTRAINT order_intent_events_entity_sequence_key UNIQUE (entity_id, sequence),
    CONSTRAINT order_intent_events_entity_dedup_key UNIQUE (entity_id, dedup_key),
    CONSTRAINT order_intent_events_event_entity_key UNIQUE (event_id, entity_id),
    CONSTRAINT order_intent_events_corrects_same_entity
        FOREIGN KEY (corrects_event_id, entity_id) REFERENCES order_intent_events (event_id, entity_id)
);

-- Broker order identity. Intent/decision/leg associations are link events, established only
-- with verified broker evidence.
CREATE TABLE orders (
    order_id         uuid        PRIMARY KEY,
    account_scope_id text        NOT NULL CHECK (length(account_scope_id) > 0),
    broker_order_id  text        NOT NULL CHECK (length(broker_order_id) > 0),
    first_run_id     uuid        NOT NULL REFERENCES runs (run_id),
    recorded_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT orders_account_broker_order_key UNIQUE (account_scope_id, broker_order_id)
);

CREATE TABLE order_events (
    event_id             uuid        PRIMARY KEY,
    entity_id            uuid        NOT NULL REFERENCES orders (order_id),
    run_id               uuid        NOT NULL REFERENCES runs (run_id),
    sequence             integer     NOT NULL CHECK (sequence > 0),
    event_type           text        NOT NULL CHECK (event_type IN (
                             'intent_linked', 'decision_linked', 'status_observed',
                             'fill_observed', 'cancellation_observed', 'reconciliation')),
    intent_id            uuid        REFERENCES order_intents (intent_id),
    status               text        CHECK (status IN (
                             'not_placed', 'placed', 'partially_filled', 'filled', 'cancelled',
                             'rejected', 'expired', 'unknown')),
    -- Fill observations: a verified broker execution ID when one exists (never manufactured);
    -- otherwise cumulative totals stored as observations (INTERFACES.md FillRecord).
    broker_execution_id  text,
    fill_quantity        numeric     CHECK (fill_quantity >= 0),
    fill_price           numeric     CHECK (fill_price >= 0),
    observed_at          timestamptz NOT NULL,
    source_as_of         timestamptz,
    source_tool_call_ids uuid[]      NOT NULL DEFAULT '{}',
    dedup_key            text        NOT NULL CHECK (length(dedup_key) > 0),
    payload              jsonb       NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(payload) = 'object'),
    corrects_event_id    uuid,
    recorded_at          timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT order_events_intent_link_has_intent
        CHECK (event_type <> 'intent_linked' OR intent_id IS NOT NULL),
    CONSTRAINT order_events_status_observed_has_status
        CHECK (event_type <> 'status_observed' OR status IS NOT NULL),
    CONSTRAINT order_events_fill_has_quantity
        CHECK (event_type <> 'fill_observed' OR fill_quantity IS NOT NULL),
    CONSTRAINT order_events_entity_sequence_key UNIQUE (entity_id, sequence),
    CONSTRAINT order_events_entity_dedup_key UNIQUE (entity_id, dedup_key),
    CONSTRAINT order_events_event_entity_key UNIQUE (event_id, entity_id),
    CONSTRAINT order_events_corrects_same_entity
        FOREIGN KEY (corrects_event_id, entity_id) REFERENCES order_events (event_id, entity_id)
);

-- Position lineage, stable through rolls (INTERFACES.md PositionBook).
CREATE TABLE positions (
    position_id      uuid        PRIMARY KEY,
    account_scope_id text        NOT NULL CHECK (length(account_scope_id) > 0),
    first_run_id     uuid        NOT NULL REFERENCES runs (run_id),
    recorded_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE position_events (
    event_id             uuid        PRIMARY KEY,
    entity_id            uuid        NOT NULL REFERENCES positions (position_id),
    run_id               uuid        NOT NULL REFERENCES runs (run_id),
    sequence             integer     NOT NULL CHECK (sequence > 0),
    event_type           text        NOT NULL CHECK (event_type IN (
                             'opened', 'fill_linked', 'roll', 'assignment', 'closed',
                             'gap', 'reconciliation', 'note')),
    order_id             uuid        REFERENCES orders (order_id),
    observed_at          timestamptz NOT NULL,
    source_as_of         timestamptz,
    source_tool_call_ids uuid[]      NOT NULL DEFAULT '{}',
    dedup_key            text        NOT NULL CHECK (length(dedup_key) > 0),
    payload              jsonb       NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(payload) = 'object'),
    corrects_event_id    uuid,
    recorded_at          timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT position_events_entity_sequence_key UNIQUE (entity_id, sequence),
    CONSTRAINT position_events_entity_dedup_key UNIQUE (entity_id, dedup_key),
    CONSTRAINT position_events_event_entity_key UNIQUE (event_id, entity_id),
    CONSTRAINT position_events_corrects_same_entity
        FOREIGN KEY (corrects_event_id, entity_id) REFERENCES position_events (event_id, entity_id)
);

-- Agentic workspace objects (Tier S). Ownership needs the name prefix AND this identity row.
CREATE TABLE workspace_objects (
    object_id        uuid        PRIMARY KEY,
    account_scope_id text        NOT NULL CHECK (length(account_scope_id) > 0),
    kind             text        NOT NULL CHECK (kind IN (
                         'watchlist', 'option_watchlist', 'scan', 'alert')),
    broker_id        text        NOT NULL CHECK (length(broker_id) > 0),
    first_run_id     uuid        NOT NULL REFERENCES runs (run_id),
    recorded_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT workspace_objects_account_kind_broker_key UNIQUE (account_scope_id, kind, broker_id)
);

CREATE TABLE workspace_events (
    event_id             uuid        PRIMARY KEY,
    entity_id            uuid        NOT NULL REFERENCES workspace_objects (object_id),
    run_id               uuid        NOT NULL REFERENCES runs (run_id),
    sequence             integer     NOT NULL CHECK (sequence > 0),
    event_type           text        NOT NULL CHECK (event_type IN (
                             'created', 'updated', 'deleted', 'observed', 'reconciliation')),
    before_state         jsonb,
    after_state          jsonb,
    observed_at          timestamptz NOT NULL,
    source_as_of         timestamptz,
    source_tool_call_ids uuid[]      NOT NULL DEFAULT '{}',
    dedup_key            text        NOT NULL CHECK (length(dedup_key) > 0),
    payload              jsonb       NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(payload) = 'object'),
    corrects_event_id    uuid,
    recorded_at          timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT workspace_events_entity_sequence_key UNIQUE (entity_id, sequence),
    CONSTRAINT workspace_events_entity_dedup_key UNIQUE (entity_id, dedup_key),
    CONSTRAINT workspace_events_event_entity_key UNIQUE (event_id, entity_id),
    CONSTRAINT workspace_events_corrects_same_entity
        FOREIGN KEY (corrects_event_id, entity_id) REFERENCES workspace_events (event_id, entity_id)
);

-- ---------------------------------------------------------------------------------------------
-- Evidence and outputs
-- ---------------------------------------------------------------------------------------------
CREATE TABLE citations (
    citation_id          uuid        PRIMARY KEY,
    run_id               uuid        NOT NULL REFERENCES runs (run_id),
    tool_call_id         uuid        NOT NULL REFERENCES tool_calls (tool_call_id),
    url                  text        NOT NULL CHECK (length(url) > 0),
    title                text,
    publisher            text,
    published_at         timestamptz,
    retrieved_at         timestamptz NOT NULL,
    tier                 text        NOT NULL CHECK (tier IN ('1', '2', '3')),
    excerpt              text        NOT NULL,
    corrects_citation_id uuid        REFERENCES citations (citation_id),
    recorded_at          timestamptz NOT NULL DEFAULT now()
);

-- The configured account only; values stay NULL when broker semantics are missing/ambiguous.
CREATE TABLE account_snapshots (
    snapshot_id                uuid        PRIMARY KEY,
    run_id                     uuid        NOT NULL REFERENCES runs (run_id),
    as_of                      timestamptz NOT NULL,
    retrieved_at               timestamptz NOT NULL,
    tool_call_ids              uuid[]      NOT NULL,
    account_ref                text        NOT NULL,
    agentic_verified           boolean     NOT NULL,
    account_value_usd          numeric(18, 2),
    available_settled_cash_usd numeric(18, 2),
    csp_reserved_cash_usd      numeric(18, 2),
    csp_cash_base_usd          numeric(18, 2),
    csp_cash_base_evidence_ids uuid[]      NOT NULL DEFAULT '{}',
    positions_ref              uuid        REFERENCES results (result_id),
    open_orders_ref            uuid        REFERENCES results (result_id),
    tax_lots_ref               uuid        REFERENCES results (result_id),
    reservation_evidence       jsonb       NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(reservation_evidence) = 'array'),
    quality                    text        NOT NULL CHECK (quality IN (
                                   'ok', 'stale', 'missing', 'contradictory', 'low_confidence')),
    gaps                       jsonb       NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(gaps) = 'array'),
    corrects_snapshot_id       uuid        REFERENCES account_snapshots (snapshot_id),
    recorded_at                timestamptz NOT NULL DEFAULT now(),
    CHECK (csp_cash_base_usd IS NULL OR cardinality(csp_cash_base_evidence_ids) > 0)
);

-- Raw redacted final model response, kept even when it fails to parse.
CREATE TABLE agent_outputs (
    output_id          uuid        PRIMARY KEY,
    run_id             uuid        NOT NULL REFERENCES runs (run_id),
    raw_redacted       text,
    observed_at        timestamptz NOT NULL,
    corrects_output_id uuid        REFERENCES agent_outputs (output_id),
    recorded_at        timestamptz NOT NULL DEFAULT now()
);

-- Parsed AgentDecisionOutput (model choices), separate from code-issued facts and records.
CREATE TABLE agent_decisions (
    agent_decision_id    uuid        PRIMARY KEY,
    run_id               uuid        NOT NULL REFERENCES runs (run_id),
    output_id            uuid        NOT NULL REFERENCES agent_outputs (output_id),
    schema_version       text        NOT NULL,
    parse_status         text        NOT NULL CHECK (parse_status IN ('valid', 'invalid')),
    parsed               jsonb,
    errors               jsonb       NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(errors) = 'array'),
    corrects_decision_id uuid        REFERENCES agent_decisions (agent_decision_id),
    recorded_at          timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT agent_decisions_valid_has_parsed CHECK (parse_status <> 'valid' OR parsed IS NOT NULL)
);

-- Code-issued DecisionFacts (OUTPUT_ASSEMBLY.md compute_decision_facts).
CREATE TABLE decision_facts (
    decision_facts_id uuid        PRIMARY KEY,
    run_id            uuid        NOT NULL REFERENCES runs (run_id),
    facts_ref         text        NOT NULL CHECK (length(facts_ref) > 0),
    observed_at       timestamptz NOT NULL,
    formula_versions  jsonb       NOT NULL CHECK (jsonb_typeof(formula_versions) = 'object'),
    input_hash        text        NOT NULL,
    facts             jsonb       NOT NULL CHECK (jsonb_typeof(facts) = 'object'),
    corrects_facts_id uuid        REFERENCES decision_facts (decision_facts_id),
    recorded_at       timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX decision_facts_run_ref_original_key
    ON decision_facts (run_id, facts_ref) WHERE corrects_facts_id IS NULL;

-- Code-assembled RunRecord (OUTPUT_ASSEMBLY.md assemble_run_record).
CREATE TABLE assembled_run_records (
    record_id          uuid        PRIMARY KEY,
    run_id             uuid        NOT NULL REFERENCES runs (run_id),
    schema_version     text        NOT NULL,
    assembler_version  text        NOT NULL,
    input_hash         text        NOT NULL,
    record             jsonb       NOT NULL CHECK (jsonb_typeof(record) = 'object'),
    assembled_at       timestamptz NOT NULL,
    corrects_record_id uuid        REFERENCES assembled_run_records (record_id),
    recorded_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE audit_findings (
    finding_id               uuid        PRIMARY KEY,
    run_id                   uuid        NOT NULL REFERENCES runs (run_id),
    check_id                 text        NOT NULL CHECK (check_id ~ '^V[1-7]([.:_-].+)?$'),
    outcome                  text        NOT NULL CHECK (outcome IN ('pass', 'violation', 'unverifiable')),
    effective_execution_mode text        NOT NULL CHECK (effective_execution_mode IN ('off', 'live')),
    decision_ref             text,
    leg_ref                  text,
    attempt_index            integer     CHECK (attempt_index >= 0),
    rule_key                 text,
    rule_value               jsonb,
    observed_value           jsonb,
    tool_call_ids            uuid[]      NOT NULL DEFAULT '{}',
    detail                   text        NOT NULL,
    audit_version            text        NOT NULL,
    context_hash             text        NOT NULL,
    corrects_finding_id      uuid        REFERENCES audit_findings (finding_id),
    recorded_at              timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE alerts_sent (
    alert_id        uuid        PRIMARY KEY,
    run_id          uuid        REFERENCES runs (run_id),
    alert_kind      text        NOT NULL CHECK (length(alert_kind) > 0),
    dedup_key       text        NOT NULL CHECK (length(dedup_key) > 0),
    payload         jsonb       NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    delivery_status text        NOT NULL CHECK (delivery_status IN ('sent', 'failed')),
    attempted_at    timestamptz NOT NULL,
    recorded_at     timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------------------------
-- Append-only enforcement: every ledger table rejects UPDATE, DELETE, and TRUNCATE.
-- ---------------------------------------------------------------------------------------------
-- Web search/fetch cache (ADR-0016). Append-only like every ledger table: an entry is the
-- recorded, successful result of one WebSearch or WebFetch call. Freshness is computed at read
-- time from retrieved_at and data_quality.freshness.news_max_age_seconds; nothing expires rows.

CREATE TABLE web_cache_entries (
    entry_id      uuid        PRIMARY KEY,
    run_id        uuid        NOT NULL REFERENCES runs (run_id),
    tool_call_id  uuid        NOT NULL UNIQUE REFERENCES tool_calls (tool_call_id),
    tool          text        NOT NULL CHECK (tool IN ('WebSearch', 'WebFetch')),
    -- Normalized query (WebSearch) or normalized URL (WebFetch); see domain/web_cache.py.
    cache_key     text        NOT NULL CHECK (length(cache_key) > 0),
    query_raw     text,
    url           text,
    result        jsonb       NOT NULL,
    retrieved_at  timestamptz NOT NULL,
    recorded_at   timestamptz NOT NULL DEFAULT now(),
    CHECK ((tool = 'WebSearch') = (query_raw IS NOT NULL)),
    CHECK ((tool = 'WebFetch') = (url IS NOT NULL))
);

CREATE INDEX web_cache_entries_key_idx ON web_cache_entries (tool, cache_key, retrieved_at DESC);
CREATE INDEX web_cache_entries_retrieved_idx ON web_cache_entries (retrieved_at DESC);

-- Tickers an entry is about: every whole-word, case-sensitive `[A-Z]{1,5}` token in the raw
-- WebSearch query (domain/web_cache.py `tickers_in`), written with the entry. Lookups by
-- ticker read this table, so attribution is explicit and indexed.
CREATE TABLE web_cache_entry_tickers (
    entry_id     uuid        NOT NULL REFERENCES web_cache_entries (entry_id),
    ticker       text        NOT NULL CHECK (ticker ~ '^[A-Z]{1,5}$'),
    recorded_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (entry_id, ticker)
);

CREATE INDEX web_cache_entry_tickers_ticker_idx ON web_cache_entry_tickers (ticker);

CREATE FUNCTION ledger_reject_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'ledger table % is append-only: % rejected', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'restrict_violation',
              HINT = 'record a new row that references the old one (corrects_*_id)';
END;
$$;

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'runs', 'tool_calls', 'order_intents', 'orders', 'positions', 'workspace_objects',
        'run_events', 'tool_call_events', 'order_intent_events', 'order_events',
        'position_events', 'workspace_events',
        'results', 'citations', 'account_snapshots', 'agent_outputs', 'decision_facts',
        'agent_decisions', 'assembled_run_records', 'audit_findings', 'alerts_sent',
        'web_cache_entries', 'web_cache_entry_tickers'
    ] LOOP
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I '
            'FOR EACH ROW EXECUTE FUNCTION ledger_reject_mutation()',
            t || '_append_only', t);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE TRUNCATE ON %I '
            'FOR EACH STATEMENT EXECUTE FUNCTION ledger_reject_mutation()',
            t || '_no_truncate', t);
    END LOOP;
END;
$$;
