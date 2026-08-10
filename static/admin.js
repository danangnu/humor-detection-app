document.addEventListener(
    "DOMContentLoaded",
    () => {

        const body =
            document.getElementById(
                "corrections-body"
            );

        const empty =
            document.getElementById(
                "empty-state"
            );

        const filter =
            document.getElementById(
                "status-filter"
            );

        const refreshBtn =
            document.getElementById(
                "refresh-btn"
            );

        const databasePath =
            document.getElementById(
                "database-path"
            );

        const adminMessage =
            document.getElementById(
                "admin-message"
            );

        const statTotal =
            document.getElementById(
                "stat-total"
            );

        const statPending =
            document.getElementById(
                "stat-pending"
            );

        const statApproved =
            document.getElementById(
                "stat-approved"
            );

        const statRejected =
            document.getElementById(
                "stat-rejected"
            );

        const retrainBtn =
            document.getElementById(
                "retrain-btn"
            );

        const promoteBtn =
            document.getElementById(
                "promote-btn"
            );

        const rollbackBtn =
            document.getElementById(
                "rollback-btn"
            );

        const trainingState =
            document.getElementById(
                "training-state"
            );

        const trainingMessage =
            document.getElementById(
                "training-message"
            );

        const trainingConfig =
            document.getElementById(
                "training-config"
            );

        const candidateGate =
            document.getElementById(
                "candidate-gate"
            );

        const activeModel =
            document.getElementById(
                "active-model"
            );

        const rollbackState =
            document.getElementById(
                "rollback-state"
            );

        const metricsEmpty =
            document.getElementById(
                "metrics-empty"
            );

        const metricsPanel =
            document.getElementById(
                "metrics-panel"
            );

        const metricsGrid =
            document.getElementById(
                "metrics-grid"
            );

        const retentionValue =
            document.getElementById(
                "retention-value"
            );

        const retentionDetail =
            document.getElementById(
                "retention-detail"
            );

        const gateCheckList =
            document.getElementById(
                "gate-check-list"
            );

        const historyList =
            document.getElementById(
                "history-list"
            );


        function setAdminMessage(
            text,
            type = "",
        ) {
            adminMessage.textContent =
                text;

            adminMessage.className =
                `admin-message ${type}`;
        }


        function pct(value) {
            if (
                typeof value !== "number"
                || !Number.isFinite(value)
            ) {
                return "—";
            }

            return (
                `${(value * 100).toFixed(2)}%`
            );
        }


        function pp(value) {
            if (
                typeof value !== "number"
                || !Number.isFinite(value)
            ) {
                return "—";
            }

            const sign =
                value >= 0 ? "+" : "";

            return (
                `${sign}${(value * 100).toFixed(2)} pp`
            );
        }


        async function api(
            url,
            options = {},
        ) {
            const response =
                await fetch(
                    url,
                    {
                        headers: {
                            "Content-Type":
                                "application/json",

                            ...(options.headers || {}),
                        },

                        ...options,
                    },
                );

            let data = {};

            try {
                data =
                    await response.json();
            } catch {
                data = {};
            }

            if (!response.ok) {
                throw new Error(
                    data.detail
                    || `Request failed (${response.status})`
                );
            }

            return data;
        }


        function updateCounts(
            counts,
        ) {
            statTotal.textContent =
                String(
                    counts.total ?? 0
                );

            statPending.textContent =
                String(
                    counts.pending ?? 0
                );

            statApproved.textContent =
                String(
                    counts.approved ?? 0
                );

            statRejected.textContent =
                String(
                    counts.rejected ?? 0
                );
        }


        function createSelect(
            values,
            selected,
            disabledValue = null,
        ) {
            const select =
                document.createElement(
                    "select"
                );

            values.forEach(value => {

                const option =
                    document.createElement(
                        "option"
                    );

                option.value = value;
                option.textContent = value;

                option.selected =
                    value === selected;

                option.disabled =
                    value
                    === disabledValue;

                select.appendChild(
                    option
                );
            });

            return select;
        }


        function renderCorrection(
            item,
        ) {
            const row =
                document.createElement(
                    "tr"
                );

            const corrected =
                createSelect(
                    [
                        "Humorous",
                        "Not Humorous",
                        "Ambiguous / Uncertain",
                    ],
                    item.corrected_label,
                    item.original_label,
                );

            const status =
                createSelect(
                    [
                        "pending",
                        "approved",
                        "rejected",
                    ],
                    item.status,
                );

            const saveBtn =
                document.createElement(
                    "button"
                );

            saveBtn.type = "button";
            saveBtn.className =
                "admin-button primary";
            saveBtn.textContent =
                "Save";

            const deleteBtn =
                document.createElement(
                    "button"
                );

            deleteBtn.type = "button";
            deleteBtn.className =
                "admin-button danger";
            deleteBtn.textContent =
                "Delete";

            saveBtn.addEventListener(
                "click",
                async () => {

                    saveBtn.disabled =
                        true;

                    deleteBtn.disabled =
                        true;

                    try {
                        const data =
                            await api(
                                `/admin/corrections/${item.id}`,
                                {
                                    method: "PUT",

                                    body:
                                        JSON.stringify(
                                            {
                                                corrected_label:
                                                    corrected.value,

                                                status:
                                                    status.value,
                                            },
                                        ),
                                },
                            );

                        updateCounts(
                            data.counts
                        );

                        setAdminMessage(
                            `Correction #${item.id} saved.`,
                            "success",
                        );

                        await Promise.all([
                            loadCorrections(),
                            loadStatus(),
                        ]);

                    } catch (error) {
                        setAdminMessage(
                            error.message,
                            "error",
                        );

                    } finally {
                        saveBtn.disabled =
                            false;

                        deleteBtn.disabled =
                            false;
                    }
                },
            );

            deleteBtn.addEventListener(
                "click",
                async () => {

                    if (
                        !window.confirm(
                            `Delete correction #${item.id}?`
                        )
                    ) {
                        return;
                    }

                    try {
                        const data =
                            await api(
                                `/admin/corrections/${item.id}`,
                                {
                                    method:
                                        "DELETE",
                                },
                            );

                        updateCounts(
                            data.counts
                        );

                        setAdminMessage(
                            `Correction #${item.id} deleted.`,
                            "success",
                        );

                        await Promise.all([
                            loadCorrections(),
                            loadStatus(),
                        ]);

                    } catch (error) {
                        setAdminMessage(
                            error.message,
                            "error",
                        );
                    }
                },
            );

            const cells = [
                item.id,
                item.input_text,
                item.original_label,
                pct(
                    Number(
                        item.original_score
                    )
                ),
                corrected,
                status,
                item.source,
                item.created_at,
            ];

            cells.forEach(
                (value, index) => {

                    const cell =
                        document.createElement(
                            "td"
                        );

                    if (index === 1) {
                        cell.className =
                            "admin-text";
                    }

                    if (
                        value
                        instanceof HTMLElement
                    ) {
                        cell.appendChild(
                            value
                        );
                    } else {
                        cell.textContent =
                            String(value ?? "");
                    }

                    row.appendChild(cell);
                },
            );

            const actionCell =
                document.createElement(
                    "td"
                );

            const actions =
                document.createElement(
                    "div"
                );

            actions.className =
                "admin-row-actions";

            actions.append(
                saveBtn,
                deleteBtn,
            );

            actionCell.appendChild(
                actions
            );

            row.appendChild(
                actionCell
            );

            return row;
        }


        async function loadCorrections() {
            try {
                const query =
                    filter.value
                        ? `?status=${encodeURIComponent(filter.value)}`
                        : "";

                const data =
                    await api(
                        `/admin/corrections${query}`
                    );

                body.innerHTML = "";

                updateCounts(
                    data.counts
                );

                databasePath.textContent =
                    `Database: ${data.database}`;

                empty.classList.toggle(
                    "hidden",
                    data.items.length !== 0,
                );

                data.items.forEach(
                    item => {
                        body.appendChild(
                            renderCorrection(
                                item
                            )
                        );
                    },
                );

            } catch (error) {
                setAdminMessage(
                    error.message,
                    "error",
                );
            }
        }


        function metricCard(
            title,
            active,
            candidate,
            delta,
        ) {
            const card =
                document.createElement(
                    "div"
                );

            card.className =
                "metric-card";

            const name =
                document.createElement(
                    "small"
                );

            name.textContent =
                title;

            const values =
                document.createElement(
                    "div"
                );

            values.className =
                "metric-values";

            values.innerHTML =
                `<span>Active <strong>${active}</strong></span>` +
                `<span>Candidate <strong>${candidate}</strong></span>` +
                `<span>Delta <strong>${delta}</strong></span>`;

            card.append(
                name,
                values,
            );

            return card;
        }


        function renderMetrics(
            candidate,
        ) {
            const comparison =
                candidate?.comparison;

            if (
                !comparison
                || !comparison.validation
            ) {
                metricsEmpty.classList
                    .remove("hidden");

                metricsPanel.classList
                    .add("hidden");

                return;
            }

            metricsEmpty.classList
                .add("hidden");

            metricsPanel.classList
                .remove("hidden");

            metricsGrid.innerHTML = "";
            gateCheckList.innerHTML = "";

            const active =
                comparison
                    .validation
                    .active
                    .binary_at_0_5;

            const current =
                comparison
                    .validation
                    .candidate
                    .binary_at_0_5;

            const delta =
                comparison
                    .validation
                    .delta;

            [
                [
                    "Accuracy",
                    pct(active.accuracy),
                    pct(current.accuracy),
                    pp(delta.accuracy),
                ],
                [
                    "Precision",
                    pct(active.precision),
                    pct(current.precision),
                    pp(delta.precision),
                ],
                [
                    "Recall",
                    pct(active.recall),
                    pct(current.recall),
                    pp(delta.recall),
                ],
                [
                    "F1",
                    pct(active.f1),
                    pct(current.f1),
                    pp(delta.f1),
                ],
                [
                    "ROC-AUC",
                    active.roc_auc?.toFixed(4)
                        ?? "—",

                    current.roc_auc?.toFixed(4)
                        ?? "—",

                    typeof delta.roc_auc
                        === "number"
                        ? delta.roc_auc.toFixed(4)
                        : "—",
                ],
            ].forEach(
                args => {
                    metricsGrid.appendChild(
                        metricCard(
                            ...args
                        )
                    );
                },
            );

            const retention =
                comparison
                    .correction_retention
                    || {};

            retentionValue.textContent =
                pct(
                    retention.accuracy
                );

            retentionDetail.textContent =
                `${retention.correct ?? 0} / ${retention.size ?? 0} eligible corrections retained`;

            const checks =
                comparison
                    .promotion_gate
                    ?.checks
                    || {};

            Object.entries(
                checks
            ).forEach(
                ([name, passed]) => {

                    const row =
                        document.createElement(
                            "div"
                        );

                    row.className =
                        `gate-check ${
                            passed
                                ? "pass"
                                : "fail"
                        }`;

                    const badge =
                        document.createElement(
                            "strong"
                        );

                    badge.textContent =
                        passed
                            ? "PASS"
                            : "FAIL";

                    const label =
                        document.createElement(
                            "span"
                        );

                    label.textContent =
                        name
                            .replaceAll(
                                "_",
                                " "
                            );

                    row.append(
                        badge,
                        label,
                    );

                    gateCheckList
                        .appendChild(row);
                },
            );
        }


        function renderTrainingConfig(
            configuration,
        ) {
            if (!configuration) {
                trainingConfig.textContent =
                    "";
                return;
            }

            trainingConfig.innerHTML =
                `<span>Replay <strong>${configuration.correction_repeat}</strong></span>` +
                `<span>LR <strong>${configuration.learning_rate}</strong></span>` +
                `<span>Epochs <strong>${configuration.epochs}</strong></span>` +
                `<span>Gate tolerance <strong>${(configuration.gate_tolerance * 100).toFixed(2)} pp</strong></span>` +
                `<span>Test <strong>${configuration.routine_test_policy}</strong></span>`;
        }


        async function loadStatus() {
            try {
                const data =
                    await api(
                        "/admin/status"
                    );

                const status =
                    data.retraining_status
                    || {};

                trainingState.textContent =
                    status.state
                    || "idle";

                trainingState.className =
                    (
                        status.state
                        === "failed"
                        || status.state
                        === "not_passed"
                    )
                        ? "state-error"
                        : (
                            status.state
                            === "passed"
                            || status.state
                            === "promoted"
                        )
                            ? "state-pass"
                            : "state-warn";

                trainingMessage.textContent =
                    status.message
                    || "";

                const candidate =
                    data.candidate
                    || {};

                if (
                    candidate
                        .promotion_gate_passed
                ) {
                    candidateGate.textContent =
                        "Passed";

                    candidateGate.className =
                        "state-pass";

                } else if (
                    candidate.metrics_available
                ) {
                    candidateGate.textContent =
                        "Not passed";

                    candidateGate.className =
                        "state-error";

                } else {
                    candidateGate.textContent =
                        "Not available";

                    candidateGate.className =
                        "state-warn";
                }

                activeModel.textContent =
                    data.active_model
                        ?.model_version
                    || "Unknown";

                rollbackState.textContent =
                    data.rollback_available
                        ? "Available"
                        : "Not available";

                rollbackState.className =
                    data.rollback_available
                        ? "state-pass"
                        : "state-warn";

                retrainBtn.disabled =
                    data.retraining_running
                    || !data.retraining_available;

                promoteBtn.disabled =
                    !candidate
                        .promotion_gate_passed
                    || candidate.status
                        === "promoted"
                    || data.retraining_running;

                rollbackBtn.disabled =
                    !data.rollback_available
                    || data.retraining_running;

                renderMetrics(
                    candidate
                );

                renderTrainingConfig(
                    data.configuration
                );

            } catch (error) {
                trainingState.textContent =
                    "Error";

                trainingMessage.textContent =
                    error.message;
            }
        }


        async function loadHistory() {
            try {
                const data =
                    await api(
                        "/admin/history"
                    );

                const items =
                    data.items || [];

                historyList.innerHTML =
                    "";

                if (!items.length) {
                    historyList.textContent =
                        "No promotions recorded.";

                    return;
                }

                [...items]
                    .reverse()
                    .forEach(item => {

                        const card =
                            document.createElement(
                                "div"
                            );

                        card.className =
                            "history-item";

                        const title =
                            document.createElement(
                                "strong"
                            );

                        title.textContent =
                            item.candidate_id;

                        const details =
                            document.createElement(
                                "span"
                            );

                        details.textContent =
                            item.rolled_back
                                ? `Promoted ${item.promoted_at_utc} · Rolled back ${item.rolled_back_at_utc}`
                                : `Promoted ${item.promoted_at_utc}`;

                        card.append(
                            title,
                            details,
                        );

                        historyList
                            .appendChild(card);
                    });

            } catch (error) {
                historyList.textContent =
                    error.message;
            }
        }


        async function startRetraining() {
            if (
                !window.confirm(
                    "Start candidate retraining using the currently approved corrections? The active model will remain unchanged."
                )
            ) {
                return;
            }

            retrainBtn.disabled = true;

            try {
                const data =
                    await api(
                        "/admin/retrain",
                        {
                            method: "POST",
                            body: "{}",
                        },
                    );

                setAdminMessage(
                    data.message,
                    "success",
                );

                await loadStatus();

            } catch (error) {
                setAdminMessage(
                    error.message,
                    "error",
                );

                await loadStatus();
            }
        }


        async function promote() {
            if (
                !window.confirm(
                    "Promote this passing candidate? The current active model will be backed up first. Humor Bot must be restarted afterward."
                )
            ) {
                return;
            }

            try {
                const data =
                    await api(
                        "/admin/promote",
                        {
                            method: "POST",

                            body:
                                JSON.stringify(
                                    {
                                        confirmation:
                                            "PROMOTE",
                                    },
                                ),
                        },
                    );

                setAdminMessage(
                    data.message,
                    "success",
                );

                await Promise.all([
                    loadStatus(),
                    loadHistory(),
                ]);

            } catch (error) {
                setAdminMessage(
                    error.message,
                    "error",
                );
            }
        }


        async function rollback() {
            if (
                !window.confirm(
                    "Restore the previous active model from the most recent promotion backup? Humor Bot must be restarted afterward."
                )
            ) {
                return;
            }

            try {
                const data =
                    await api(
                        "/admin/rollback",
                        {
                            method: "POST",

                            body:
                                JSON.stringify(
                                    {
                                        confirmation:
                                            "ROLLBACK",
                                    },
                                ),
                        },
                    );

                setAdminMessage(
                    data.message,
                    "success",
                );

                await Promise.all([
                    loadStatus(),
                    loadHistory(),
                ]);

            } catch (error) {
                setAdminMessage(
                    error.message,
                    "error",
                );
            }
        }


        async function refreshAll() {
            refreshBtn.disabled =
                true;

            try {
                await Promise.all([
                    loadCorrections(),
                    loadStatus(),
                    loadHistory(),
                ]);

                setAdminMessage(
                    "Administration data refreshed."
                );

            } finally {
                refreshBtn.disabled =
                    false;
            }
        }


        filter.addEventListener(
            "change",
            loadCorrections,
        );

        refreshBtn.addEventListener(
            "click",
            refreshAll,
        );

        retrainBtn.addEventListener(
            "click",
            startRetraining,
        );

        promoteBtn.addEventListener(
            "click",
            promote,
        );

        rollbackBtn.addEventListener(
            "click",
            rollback,
        );


        refreshAll();

        // Training status and candidate metrics are polled while
        // the page remains open.
        setInterval(
            async () => {
                await loadStatus();
            },
            5000,
        );
    },
);
