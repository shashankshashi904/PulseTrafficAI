const API_BASE = "http://127.0.0.1:8000";

let trafficCharts = {};
let metadata = null;

document.addEventListener("DOMContentLoaded", async () => {
    setDefaultDateTime();
    setupEventListeners();
    updateIncidentUI();
    await loadMetadata();
});

function setDefaultDateTime() {
    const now = new Date();

    const dateInput =
        document.getElementById("predictionDate");

    const timeInput =
        document.getElementById("predictionTime");

    if (dateInput) {
        dateInput.value =
            formatDateInput(now);
    }

    if (timeInput) {
        timeInput.value =
            formatTimeInput(now);
    }
}

function formatDateInput(date) {
    const year =
        date.getFullYear();

    const month =
        String(date.getMonth() + 1)
            .padStart(2, "0");

    const day =
        String(date.getDate())
            .padStart(2, "0");

    return `${year}-${month}-${day}`;
}

function formatTimeInput(date) {
    const hours =
        String(date.getHours())
            .padStart(2, "0");

    const minutes =
        String(date.getMinutes())
            .padStart(2, "0");

    return `${hours}:${minutes}`;
}

function setupEventListeners() {
    const locationSelect =
        document.getElementById("location");

    const streetSelect =
        document.getElementById("street");

    const sensorSelect =
        document.getElementById("sensor");

    const incidentCheckbox =
        document.getElementById("incidentEnabled");

    const predictButton =
        document.getElementById("predictButton");

    if (locationSelect) {
        locationSelect.addEventListener(
            "change",
            loadMetadata
        );
    }

    if (streetSelect) {
        streetSelect.addEventListener(
            "change",
            populateSensors
        );
    }

    if (sensorSelect) {
        sensorSelect.addEventListener(
            "change",
            updateSelectedSensorInfo
        );
    }

    if (incidentCheckbox) {
        incidentCheckbox.addEventListener(
            "change",
            updateIncidentUI
        );
    }

    if (predictButton) {
        predictButton.addEventListener(
            "click",
            predictTraffic
        );
    }
}

function updateIncidentUI() {
    const incidentCheckbox =
        document.getElementById(
            "incidentEnabled"
        );

    const incidentWarning =
        document.getElementById(
            "incidentInfoBox"
        );

    const incidentResultBox =
        document.getElementById(
            "incidentResultBox"
        );

    const incidentEnabled =
        incidentCheckbox?.checked || false;

    /*
     * Input-area warning
     *
     * This message appears immediately when
     * incident-aware prediction is enabled.
     */
    if (incidentWarning) {
        incidentWarning.style.display =
            incidentEnabled
                ? "block"
                : "none";
    }

    /*
     * If incident mode is switched OFF,
     * remove the previous prediction's
     * incident information from the result area.
     */
    if (
        !incidentEnabled &&
        incidentResultBox
    ) {
        incidentResultBox.style.display =
            "none";
    }
}

async function loadMetadata() {
    const locationSelect =
        document.getElementById("location");

    if (!locationSelect) {
        return;
    }

    const location =
        locationSelect.value;

    try {
        setText(
            "sensorInfo",
            "Loading..."
        );

        const response =
            await fetch(
                `${API_BASE}/api/v1/metadata?location=${encodeURIComponent(location)}`
            );

        if (!response.ok) {
            throw new Error(
                `Metadata request failed: ${response.status}`
            );
        }

        metadata =
            await response.json();

        populateStreets();

    } catch (error) {
        console.error(
            "Metadata error:",
            error
        );

        showError(
            "Unable to load road and sensor information. " +
            "Make sure the FastAPI backend is running."
        );
    }
}

function populateStreets() {
    const streetSelect =
        document.getElementById("street");

    if (!streetSelect || !metadata) {
        return;
    }

    streetSelect.innerHTML = "";

    const streets =
        metadata.streets || [];

    if (streets.length === 0) {
        const option =
            document.createElement("option");

        option.value = "";

        option.textContent =
            "No roads available";

        streetSelect.appendChild(
            option
        );

        return;
    }

    streets.forEach(street => {
        const option =
            document.createElement("option");

        option.value =
            street;

        option.textContent =
            street;

        streetSelect.appendChild(
            option
        );
    });

    populateSensors();
}

function populateSensors() {
    const streetSelect =
        document.getElementById("street");

    const sensorSelect =
        document.getElementById("sensor");

    if (
        !streetSelect ||
        !sensorSelect ||
        !metadata
    ) {
        return;
    }

    const selectedStreet =
        streetSelect.value;

    sensorSelect.innerHTML = "";

    const sensors =
        (metadata.sensors || [])
            .filter(
                sensor =>
                    sensor.street ===
                    selectedStreet
            );

    if (sensors.length === 0) {
        const option =
            document.createElement("option");

        option.value = "";

        option.textContent =
            "No sensors available";

        sensorSelect.appendChild(
            option
        );

        return;
    }

    sensors.forEach(sensor => {
        const option =
            document.createElement("option");

        option.value =
            sensor.node;

        option.textContent =
            `${sensor.node} - ${sensor.name} (${sensor.station_id})`;

        sensorSelect.appendChild(
            option
        );
    });

    updateSelectedSensorInfo();
}

function updateSelectedSensorInfo() {
    const sensorSelect =
        document.getElementById("sensor");

    if (
        !sensorSelect ||
        !metadata
    ) {
        return;
    }

    const node =
        Number(
            sensorSelect.value
        );

    const sensor =
        (metadata.sensors || [])
            .find(
                item =>
                    Number(item.node) ===
                    node
            );

    if (!sensor) {
        return;
    }

    setText(
        "sensorInfo",
        `${sensor.name} | Station ${sensor.station_id}`
    );
}

async function predictTraffic() {

    /*
     * Hide the previous incident information
     * before starting a new prediction.
     *
     * This prevents old incident information
     * from remaining visible while the new
     * prediction is being generated.
     */
    const incidentResultBox =
        document.getElementById(
            "incidentResultBox"
        );

    if (incidentResultBox) {
        incidentResultBox.style.display =
            "none";
    }

    const date =
        document.getElementById(
            "predictionDate"
        )?.value;

    const time =
        document.getElementById(
            "predictionTime"
        )?.value;

    const location =
        document.getElementById(
            "location"
        )?.value;

    const street =
        document.getElementById(
            "street"
        )?.value;

    const sensor =
        document.getElementById(
            "sensor"
        )?.value;

    const incidentEnabled =
        document.getElementById(
            "incidentEnabled"
        )?.checked || false;

    if (
        !date ||
        !time ||
        !location ||
        !street ||
        !sensor
    ) {
        showError(
            "Please select Date, Time, Region, Road and Sensor."
        );

        return;
    }

    const requestBody = {
        date: date,
        time: time,
        location: location,
        street: street,
        sensor: Number(sensor),
        incident_enabled:
            incidentEnabled,
        incident_time: null,
        incident_description: null,
        incident_type: null,
        holiday: null
    };

    setPredictButtonState(true);

    hideError();

    try {
        console.log(
            "Prediction request:",
            requestBody
        );

        const response =
            await fetch(
                `${API_BASE}/api/v1/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            requestBody
                        )
                }
            );

        const result =
            await response.json();

        if (!response.ok) {
            throw new Error(
                result.detail ||
                `Prediction failed: ${response.status}`
            );
        }

        console.log(
            "Prediction response:",
            result
        );

        displayPrediction(result);

    } catch (error) {
        console.error(
            "Prediction error:",
            error
        );

        showError(
            error.message ||
            "Prediction failed."
        );

    } finally {
        setPredictButtonState(false);
    }
}

function displayPrediction(result) {
    hideError();

    const incidentEnabled =
        result.mode ===
        "Incident-Aware Traffic";

    updateIncidentResultInformation(
        result,
        incidentEnabled
    );

    setText(
        "modelName",
        result.model || "IGSTGNN"
    );

    setText(
        "modelMode",
        result.mode || "Normal Traffic"
    );

    const predictionStart =
        formatPredictionStart(
            result.prediction_start
        );

    const selectedSensor =
        result.sensors?.find(
            sensor =>
                sensor.role ===
                "selected"
        );

    if (!selectedSensor) {
        showError(
            "Selected sensor forecast was not returned."
        );

        return;
    }

    const forecastValues =
        selectedSensor.forecast.map(
            point =>
                Number(point.traffic)
        );

    const peakTraffic =
        Math.max(
            ...forecastValues
        );

    const averageTraffic =
        forecastValues.reduce(
            (sum, value) =>
                sum + value,
            0
        ) /
        forecastValues.length;

    setText(
        "selectedSensorSummary",
        `${selectedSensor.sensor_name} (${selectedSensor.station_id})`
    );

    setText(
        "peakTraffic",
        formatTrafficValue(
            peakTraffic
        )
    );

    setText(
        "averageTraffic",
        formatTrafficValue(
            averageTraffic
        )
    );

    setText(
        "trafficStatus",
        getTrafficStatus(
            peakTraffic
        )
    );

    const chartDefinitions = [
        {
            role: "previous",

            canvasId:
                "previousSensorChart",

            infoId:
                "previousSensorInfo",

            startId:
                "previousPredictionStart",

            cardId:
                "previousChartCard",

            incident: false
        },

        {
            role: "selected",

            canvasId:
                "selectedSensorChart",

            infoId:
                "selectedSensorInfo",

            startId:
                "selectedPredictionStart",

            cardId:
                "selectedChartCard",

            incident:
                incidentEnabled
        },

        {
            role: "next",

            canvasId:
                "nextSensorChart",

            infoId:
                "nextSensorInfo",

            startId:
                "nextPredictionStart",

            cardId:
                "nextChartCard",

            incident: false
        }
    ];

    chartDefinitions.forEach(
        definition => {

            const sensorData =
                result.sensors?.find(
                    sensor =>
                        sensor.role ===
                        definition.role
                );

            if (!sensorData) {
                return;
            }

            setText(
                definition.infoId,
                `Sensor ${sensorData.sensor_node} | ` +
                `Station ${sensorData.station_id}`
            );

            setText(
                definition.startId,
                predictionStart
            );

            createTrafficChart(
                definition.canvasId,
                sensorData,
                definition.incident
            );

            const card =
                document.getElementById(
                    definition.cardId
                );

            if (card) {
                card.classList.toggle(
                    "incident-active",
                    definition.incident
                );
            }
        }
    );

    populateForecastTable(
        selectedSensor
    );

    const resultsSection =
        document.getElementById(
            "predictionResults"
        );

    if (resultsSection) {

        resultsSection.style.display =
            "block";

        resultsSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }
}

function updateIncidentResultInformation(
    result,
    incidentEnabled
) {
    const box =
        document.getElementById(
            "incidentResultBox"
        );

    if (!box) {
        return;
    }

    /*
     * Normal Traffic mode:
     * incident information must remain hidden.
     */
    if (!incidentEnabled) {
        box.style.display =
            "none";

        return;
    }

    const incidentInformation =
        result.incident_information;

    /*
     * Incident mode is enabled but backend
     * did not return incident information.
     */
    if (!incidentInformation) {
        box.style.display =
            "none";

        return;
    }

    /*
     * Display only the three UI fields
     * that we decided to expose:
     *
     * 1. Incident Type
     * 2. Description
     * 3. Holiday
     *
     * Incident Time, Position and Duration
     * remain hidden from the UI.
     */
    setText(
        "incidentResultType",
        incidentInformation.incident_type ||
        "Unknown"
    );

    setText(
        "incidentResultDescription",
        incidentInformation.description ||
        "Unknown"
    );

    setText(
        "incidentResultHoliday",
        incidentInformation.is_holiday
            ? "Yes"
            : "No"
    );

    box.style.display =
        "grid";
}

function createTrafficChart(
    canvasId,
    sensorData,
    incidentMode
) {
    const canvas =
        document.getElementById(
            canvasId
        );

    if (!canvas) {
        return;
    }

    if (trafficCharts[canvasId]) {
        trafficCharts[
            canvasId
        ].destroy();
    }

    const historical =
        sensorData.historical || [];

    const forecast =
        sensorData.forecast || [];

    const labels = [
        ...historical.map(
            point =>
                point.time
        ),

        ...forecast.map(
            point =>
                point.time
        )
    ];

    const historicalValues = [
        ...historical.map(
            point =>
                Number(point.traffic)
        ),

        ...Array(
            forecast.length
        ).fill(null)
    ];

    const forecastValues = [
        ...Array(
            historical.length
        ).fill(null),

        ...forecast.map(
            point =>
                Number(point.traffic)
        )
    ];

    const forecastColor =
        incidentMode
            ? "#ef4444"
            : "#f97316";

    const forecastBackground =
        incidentMode
            ? "rgba(239, 68, 68, 0.10)"
            : "rgba(249, 115, 22, 0.10)";

    trafficCharts[canvasId] =
        new Chart(
            canvas.getContext("2d"),
            {
                plugins: [
                    {
                        id:
                            "predictionStartLine",

                        afterDraw(chart) {

                            if (
                                historical.length === 0 ||
                                forecast.length === 0
                            ) {
                                return;
                            }

                            const xScale =
                                chart.scales.x;

                            const chartArea =
                                chart.chartArea;

                            if (
                                !xScale ||
                                !chartArea
                            ) {
                                return;
                            }

                            const lastHistoricalIndex =
                                historical.length - 1;

                            const firstForecastIndex =
                                historical.length;

                            const lastHistoricalX =
                                xScale.getPixelForValue(
                                    lastHistoricalIndex
                                );

                            const firstForecastX =
                                xScale.getPixelForValue(
                                    firstForecastIndex
                                );

                            const predictionStartX =
                                (
                                    lastHistoricalX +
                                    firstForecastX
                                ) / 2;

                            const ctx =
                                chart.ctx;

                            ctx.save();

                            ctx.beginPath();

                            ctx.setLineDash([
                                4,
                                4
                            ]);

                            ctx.lineWidth =
                                1.5;

                            ctx.strokeStyle =
                                "#374151";

                            ctx.moveTo(
                                predictionStartX,
                                chartArea.top
                            );

                            ctx.lineTo(
                                predictionStartX,
                                chartArea.bottom
                            );

                            ctx.stroke();

                            ctx.restore();
                        }
                    }
                ],

                type: "line",

                data: {

                    labels:
                        labels,

                    datasets: [

                        {
                            label:
                                "Historical Actual",

                            data:
                                historicalValues,

                            borderColor:
                                "#2563eb",

                            backgroundColor:
                                "rgba(37, 99, 235, 0.08)",

                            borderWidth:
                                2,

                            pointRadius:
                                2,

                            pointHoverRadius:
                                4,

                            tension:
                                0.25,

                            spanGaps:
                                false
                        },

                        {
                            label:
                                incidentMode
                                    ? "Forecast (Incident Mode)"
                                    : "Forecast (Normal Mode)",

                            data:
                                forecastValues,

                            borderColor:
                                forecastColor,

                            backgroundColor:
                                forecastBackground,

                            borderWidth:
                                2.5,

                            pointRadius:
                                2.5,

                            pointHoverRadius:
                                4,

                            tension:
                                0.25,

                            spanGaps:
                                false
                        }
                    ]
                },

                options: {

                    responsive:
                        true,

                    maintainAspectRatio:
                        false,

                    interaction: {

                        mode:
                            "index",

                        intersect:
                            false
                    },

                    plugins: {

                        legend: {

                            display:
                                false
                        },

                        tooltip: {

                            callbacks: {

                                label:
                                    function(context) {

                                        if (
                                            context.raw === null ||
                                            context.raw === undefined
                                        ) {
                                            return "";
                                        }

                                        return (
                                            `${context.dataset.label}: ` +
                                            `${Number(context.raw).toFixed(2)}`
                                        );
                                    }
                            }
                        }
                    },

                    scales: {

                        x: {

                            ticks: {

                                maxTicksLimit:
                                    7,

                                font: {
                                    size:
                                        10
                                }
                            },

                            grid: {

                                color:
                                    "rgba(148, 163, 184, 0.18)"
                            }
                        },

                        y: {

                            beginAtZero:
                                false,

                            ticks: {

                                font: {
                                    size:
                                        10
                                }
                            },

                            title: {

                                display:
                                    true,

                                text:
                                    "Traffic",

                                font: {
                                    size:
                                        10
                                }
                            },

                            grid: {

                                color:
                                    "rgba(148, 163, 184, 0.18)"
                            }
                        }
                    }
                }
            }
        );
}

function populateForecastTable(
    sensorData
) {
    const tableBody =
        document.getElementById(
            "forecastTableBody"
        );

    if (!tableBody) {
        return;
    }

    tableBody.innerHTML = "";

    (
        sensorData.forecast || []
    ).forEach(
        (point, index) => {

            const traffic =
                Number(point.traffic);

            const row =
                document.createElement(
                    "tr"
                );

            const stepCell =
                document.createElement(
                    "td"
                );

            const timeCell =
                document.createElement(
                    "td"
                );

            const trafficCell =
                document.createElement(
                    "td"
                );

            const statusCell =
                document.createElement(
                    "td"
                );

            const status =
                getTrafficStatus(
                    traffic
                );

            stepCell.textContent =
                `T+${(index + 1) * 5} min`;

            timeCell.textContent =
                point.time;

            trafficCell.textContent =
                traffic.toFixed(2);

            statusCell.textContent =
                status;

            statusCell.className =
                `status-${status
                    .toLowerCase()
                    .replace(" ", "-")}`;

            row.appendChild(
                stepCell
            );

            row.appendChild(
                timeCell
            );

            row.appendChild(
                trafficCell
            );

            row.appendChild(
                statusCell
            );

            tableBody.appendChild(
                row
            );
        }
    );
}

function getTrafficStatus(value) {
    if (value < 100) {
        return "Low";
    }

    if (value < 200) {
        return "Moderate";
    }

    if (value < 300) {
        return "High";
    }

    return "Very High";
}

function formatTrafficValue(value) {
    if (!Number.isFinite(value)) {
        return "--";
    }

    return value.toFixed(2);
}

function formatPredictionStart(value) {
    if (!value) {
        return "--";
    }

    const date =
        new Date(value);

    if (
        Number.isNaN(
            date.getTime()
        )
    ) {
        return value;
    }

    return date.toLocaleTimeString(
        [],
        {
            hour:
                "2-digit",

            minute:
                "2-digit"
        }
    );
}

function setText(
    id,
    value
) {
    const element =
        document.getElementById(id);

    if (element) {
        element.textContent =
            value;
    }
}

function setPredictButtonState(
    isLoading
) {
    const button =
        document.getElementById(
            "predictButton"
        );

    if (!button) {
        return;
    }

    button.disabled =
        isLoading;

    button.textContent =
        isLoading
            ? "Predicting..."
            : "Predict Traffic";
}

function showError(
    message
) {
    const element =
        document.getElementById(
            "errorMessage"
        );

    if (!element) {
        console.error(
            message
        );

        return;
    }

    element.textContent =
        message;

    element.style.display =
        "block";
}

function hideError() {
    const element =
        document.getElementById(
            "errorMessage"
        );

    if (element) {
        element.style.display =
            "none";
    }
}