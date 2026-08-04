document.addEventListener("DOMContentLoaded", () => {
    // ------------------------------------------------------------
    // DOM elements
    // ------------------------------------------------------------

    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");

    const analyzeBtn = document.getElementById("analyze-btn");
    const sendBtn = document.getElementById("send-btn");
    const clearBtn = document.getElementById("clear-btn");

    const charCount = document.getElementById("char-count");

    const apiStatus = document.getElementById("api-status");
    const modelStatus = document.getElementById("model-status");
    const geminiStatus = document.getElementById("gemini-status");

    const lowBand = document.getElementById("low-band");
    const highBand = document.getElementById("high-band");

    let busy = false;


    // ------------------------------------------------------------
    // Formatting helpers
    // ------------------------------------------------------------

    function formatPercent(value) {
        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "—";
        }

        const percentage = number * 100;

        // Show a little more detail for scores that are near 0 or 100.
        if (percentage > 0 && percentage < 1) {
            return `${percentage.toFixed(1)}%`;
        }

        if (percentage > 99 && percentage < 100) {
            return `${percentage.toFixed(1)}%`;
        }

        return `${Math.round(percentage)}%`;
    }


    function getCurrentText() {
        return userInput.value.trim();
    }


    function updateCounter() {
        charCount.textContent = `${userInput.value.length} / 8000`;
    }


    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatBox.scrollTop = chatBox.scrollHeight;
        });
    }


    // ------------------------------------------------------------
    // Busy state
    // ------------------------------------------------------------

    function setBusy(value) {
        busy = value;

        analyzeBtn.disabled = value;
        sendBtn.disabled = value;

        analyzeBtn.setAttribute("aria-busy", value ? "true" : "false");
        sendBtn.setAttribute("aria-busy", value ? "true" : "false");
    }


    // ------------------------------------------------------------
    // Result normalization
    // ------------------------------------------------------------

    function normalizeResult(result) {
        if (!result) {
            return null;
        }

        return {
            score:
                result.score ??
                result.user_humor_score ??
                null,

            label:
                result.label ??
                result.user_label ??
                "Unknown",

            decisionStrength:
                result.confidence ??
                result.user_confidence ??
                "—",

            explanation:
                result.explanation ??
                result.user_explanation ??
                "",

            modelSource:
                result.model_source ??
                "",

            lowThreshold:
                result.low_threshold ??
                null,

            highThreshold:
                result.high_threshold ??
                null
        };
    }


    // ------------------------------------------------------------
    // Message rendering
    // ------------------------------------------------------------

    function appendMessage(
        text,
        type,
        result = null,
        responseSource = ""
    ) {
        const wrapper = document.createElement("div");
        wrapper.className = `message ${type}`;

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = text;

        wrapper.appendChild(bubble);


        // --------------------------------------------------------
        // Model result card
        // --------------------------------------------------------

        if (result) {
            const normalized = normalizeResult(result);

            const card = document.createElement("div");
            card.className = "result-card";


            const rows = [
                [
                    "Humor score",
                    formatPercent(normalized.score)
                ],
                [
                    "Classification",
                    normalized.label
                ],
                [
                    "Decision strength",
                    normalized.decisionStrength
                ]
            ];


            rows.forEach(([name, value]) => {
                const row = document.createElement("div");
                row.className = "result-row";

                const nameElement = document.createElement("span");
                const valueElement = document.createElement("strong");

                nameElement.textContent = name;
                valueElement.textContent = value;

                row.append(nameElement, valueElement);
                card.appendChild(row);
            });


            // ----------------------------------------------------
            // Explanation
            // ----------------------------------------------------

            if (normalized.explanation) {
                const explanation = document.createElement("p");

                explanation.className = "explanation";
                explanation.textContent = normalized.explanation;

                card.appendChild(explanation);
            }


            // ----------------------------------------------------
            // Threshold information
            // ----------------------------------------------------

            if (
                normalized.lowThreshold !== null &&
                normalized.highThreshold !== null
            ) {
                const thresholdInfo = document.createElement("p");

                thresholdInfo.className = "explanation threshold-info";

                thresholdInfo.textContent =
                    `Current decision range: below ` +
                    `${formatPercent(normalized.lowThreshold)} = Not Humorous, ` +
                    `${formatPercent(normalized.lowThreshold)}–` +
                    `${formatPercent(normalized.highThreshold)} = Ambiguous, ` +
                    `${formatPercent(normalized.highThreshold)}+ = Humorous.`;

                card.appendChild(thresholdInfo);
            }


            // ----------------------------------------------------
            // Source information
            // ----------------------------------------------------

            const sourceParts = [];

            if (responseSource) {
                sourceParts.push(responseSource);
            }

            if (normalized.modelSource) {
                sourceParts.push(
                    `Model: ${normalized.modelSource}`
                );
            }

            if (sourceParts.length > 0) {
                const sourceElement = document.createElement("div");

                sourceElement.className = "source";
                sourceElement.textContent = sourceParts.join(" · ");

                card.appendChild(sourceElement);
            }


            wrapper.appendChild(card);
        }


        chatBox.appendChild(wrapper);
        scrollToBottom();
    }


    // ------------------------------------------------------------
    // Loading indicator
    // ------------------------------------------------------------

    function showTyping(text) {
        removeTyping();

        const element = document.createElement("div");

        element.id = "typing";
        element.className = "typing";
        element.textContent = `${text}…`;

        chatBox.appendChild(element);

        scrollToBottom();
    }


    function removeTyping() {
        document.getElementById("typing")?.remove();
    }


    // ------------------------------------------------------------
    // API helper
    // ------------------------------------------------------------

    async function request(url, body) {
        let response;

        try {
            response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(body)
            });
        } catch (error) {
            throw new Error(
                "Could not connect to the Humor Bot API."
            );
        }


        let data = {};

        try {
            data = await response.json();
        } catch {
            data = {};
        }


        if (!response.ok) {
            let message =
                data.detail ||
                data.message ||
                `Request failed (${response.status})`;

            if (typeof message === "object") {
                message = JSON.stringify(message);
            }

            throw new Error(message);
        }


        return data;
    }


    // ------------------------------------------------------------
    // Health check
    // ------------------------------------------------------------

    async function loadHealth() {
        try {
            const response = await fetch("/health");

            if (!response.ok) {
                throw new Error(
                    `Health check failed (${response.status})`
                );
            }

            const data = await response.json();


            // API
            apiStatus.textContent = "Online";


            // Model
            if (data.local_model_ready) {
                modelStatus.textContent = "Local model";
            } else if (data.bootstrap_allowed) {
                modelStatus.textContent = "Bootstrap ready";
            } else {
                modelStatus.textContent = "Not trained";
            }


            // Gemini
            geminiStatus.textContent =
                data.gemini_configured
                    ? "Configured"
                    : "Fallback";


            // Threshold display
            if (data.thresholds) {
                if (
                    data.thresholds.low !== undefined &&
                    data.thresholds.low !== null
                ) {
                    lowBand.textContent =
                        formatPercent(data.thresholds.low);
                }

                if (
                    data.thresholds.high !== undefined &&
                    data.thresholds.high !== null
                ) {
                    highBand.textContent =
                        formatPercent(data.thresholds.high);
                }
            }

        } catch (error) {
            apiStatus.textContent = "Offline";
            modelStatus.textContent = "Unknown";
            geminiStatus.textContent = "Unknown";
        }
    }


    // ------------------------------------------------------------
    // Analyze only
    // ------------------------------------------------------------

    async function analyze() {
        const text = getCurrentText();

        if (!text || busy) {
            return;
        }


        appendMessage(
            text,
            "user"
        );


        userInput.value = "";
        updateCounter();

        setBusy(true);
        showTyping("Analyzing");


        try {
            const data = await request(
                "/analyze",
                {
                    text: text
                }
            );


            removeTyping();


            appendMessage(
                "Here is the model result.",
                "bot",
                data,
                "Analyze only"
            );

        } catch (error) {
            removeTyping();

            appendMessage(
                error.message,
                "error"
            );

        } finally {
            setBusy(false);
            userInput.focus();
        }
    }


    // ------------------------------------------------------------
    // Analyze + generate reply
    // ------------------------------------------------------------

    async function generate() {
        const text = getCurrentText();

        if (!text || busy) {
            return;
        }


        appendMessage(
            text,
            "user"
        );


        userInput.value = "";
        updateCounter();

        setBusy(true);
        showTyping("Analyzing and generating");


        try {
            const data = await request(
                "/chat",
                {
                    message: text
                }
            );


            removeTyping();


            appendMessage(
                data.reply || "No reply was returned.",
                "bot",
                data,
                data.response_source || ""
            );

        } catch (error) {
            removeTyping();

            appendMessage(
                error.message,
                "error"
            );

        } finally {
            setBusy(false);
            userInput.focus();
        }
    }


    // ------------------------------------------------------------
    // Reset chat
    // ------------------------------------------------------------

    function resetChat() {
        chatBox.innerHTML = "";

        appendMessage(
            "Enter a short piece of text. I will classify it as Humorous, " +
            "Not Humorous, or Ambiguous / Uncertain.",
            "bot"
        );
    }


    // ------------------------------------------------------------
    // Events
    // ------------------------------------------------------------

    analyzeBtn.addEventListener(
        "click",
        analyze
    );


    sendBtn.addEventListener(
        "click",
        generate
    );


    clearBtn.addEventListener(
        "click",
        resetChat
    );


    userInput.addEventListener(
        "input",
        updateCounter
    );


    userInput.addEventListener(
        "keydown",
        event => {
            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {
                event.preventDefault();
                generate();
            }
        }
    );


    document
        .querySelectorAll("[data-sample]")
        .forEach(button => {

            button.addEventListener(
                "click",
                () => {
                    userInput.value =
                        button.dataset.sample || "";

                    updateCounter();
                    userInput.focus();
                }
            );

        });


    // ------------------------------------------------------------
    // Initial state
    // ------------------------------------------------------------

    resetChat();
    updateCounter();
    loadHealth();
    userInput.focus();
});