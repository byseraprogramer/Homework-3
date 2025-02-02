import React, { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
    zoomPlugin
);

const App = () => {
    const [companies, setCompanies] = useState([]);
    const [selectedCompany, setSelectedCompany] = useState("");
    const [transactions, setTransactions] = useState([]);
    const [filteredTransactions, setFilteredTransactions] = useState([]);
    const [predictions, setPredictions] = useState([]);
    const [fromDate, setFromDate] = useState("");
    const [toDate, setToDate] = useState("");

    useEffect(() => {
        fetch("http://localhost:8080/api/companies")
            .then((response) => response.json())
            .then((data) => setCompanies(data));
    }, []);

    const handleFetchTransactions = () => {
        if (selectedCompany) {
            setPredictions([]);
            fetch(`http://localhost:8080/api/transactions/${selectedCompany}`)
                .then((response) => response.json())
                .then((data) => {
                    const sortedTransactions = data.sort(
                        (a, b) => new Date(a.date) - new Date(b.date)
                    );
                    const filtered = fromDate && toDate
                        ? sortedTransactions.filter((transaction) => {
                            const transactionDate = new Date(transaction.date);
                            return (
                                transactionDate >= new Date(fromDate) &&
                                transactionDate <= new Date(toDate)
                            );
                        })
                        : sortedTransactions;
                    setFilteredTransactions(filtered);
                    setTransactions(sortedTransactions);
                    fetch(`http://localhost:8080/api/predict/${selectedCompany}`)
                        .then((response) => response.json())
                        .then((data) => setPredictions(data))
                        .catch((error) =>
                            console.error("Error fetching predictions:", error)
                        );
                });
        } else {
            alert("Please select a company.");
        }
    };

    const handleReset = () => {
        setFromDate("");
        setToDate("");
        setFilteredTransactions(transactions);
        setPredictions([]);
    };

    const formatDate = (dateInput) => {
        const d = new Date(dateInput);
        return `${d.getDate().toString().padStart(2, "0")}.${(d.getMonth() + 1)
            .toString()
            .padStart(2, "0")}.${d.getFullYear()}`;
    };

    const transactionLabels = filteredTransactions.map((transaction) =>
        formatDate(transaction.date)
    );

    let predictedLabels = [];
    if (filteredTransactions.length > 0 && predictions.length > 0) {
        const lastDate = new Date(
            filteredTransactions[filteredTransactions.length - 1].date
        );
        predictedLabels = predictions.map((_, index) => {
            const d = new Date(lastDate);
            d.setDate(d.getDate() + index + 1);
            return formatDate(d);
        });
    }
    const labels = [...transactionLabels, ...predictedLabels];

    const padData = (dataArray) => [
        ...dataArray,
        ...new Array(predictedLabels.length).fill(null),
    ];

    let predColor = "red";
    if (predictions.length > 0 && filteredTransactions.length > 0) {
        const lastPrice = parseFloat(
            filteredTransactions[filteredTransactions.length - 1].lastPrice
        );
        const firstPrediction = parseFloat(predictions[0]);
        if (firstPrediction > lastPrice) {
            predColor = "green";
        }
    }

    // Map signal values to colors
    const getSignalColor = (signal) => {
        if (signal === "Buy") return "green";  // 🟢
        if (signal === "Sell") return "red";   // 🔴
        return "blue";                         // 🔵 (Hold)
    };

    const chartData = {
        labels,
        datasets: [
            { label: "SMA (10)", data: padData(filteredTransactions.map((t) => parseFloat(t.sma10))), borderColor: "blue", borderWidth: 2, pointRadius: 2, fill: false },
            { label: "SMA (20)", data: padData(filteredTransactions.map((t) => parseFloat(t.sma_20))), borderColor: "lightblue", borderWidth: 2, pointRadius: 2, fill: false },
            { label: "EMA (10)", data: padData(filteredTransactions.map((t) => parseFloat(t.ema10))), borderColor: "purple", borderWidth: 2, pointRadius: 2, fill: false },
            { label: "EMA (20)", data: padData(filteredTransactions.map((t) => parseFloat(t.ema_20))), borderColor: "violet", borderWidth: 2, pointRadius: 2, fill: false },
            { label: "WMA (10)", data: padData(filteredTransactions.map((t) => parseFloat(t.wma_10))), borderColor: "brown", borderWidth: 2, pointRadius: 2, fill: false },
            { label: "RSI", data: padData(filteredTransactions.map((t) => parseFloat(t.rsi))), borderColor: "orange", borderWidth: 2, pointRadius: 2, fill: false },
            { label: "CCI", data: padData(filteredTransactions.map((t) => parseFloat(t.cci))), borderColor: "pink", borderWidth: 2, pointRadius: 2, fill: false },
            { label: "Stochastic K", data: padData(filteredTransactions.map((t) => parseFloat(t.k))), borderColor: "yellow", borderWidth: 2, pointRadius: 2, fill: false },
            { label: "Stochastic D", data: padData(filteredTransactions.map((t) => parseFloat(t.d))), borderColor: "cyan", borderWidth: 2, pointRadius: 2, fill: false },

            // ** Trading Signals (Plotted as Dots) **
            {
                label: "Trading Signal",
                data: padData(filteredTransactions.map((t) => parseFloat(t.lastPrice))),
                pointBackgroundColor: filteredTransactions.map((t) => getSignalColor(t.signal)),
                pointRadius: 5,
                borderColor: "transparent",
                fill: false,
                showLine: false,
            },

            {
                label: "Predicted Price",
                data: [...new Array(filteredTransactions.length).fill(null), ...predictions.map((p) => parseFloat(p))],
                borderColor: predColor,
                borderWidth: 2,
                pointRadius: 2,
                fill: false,
            },
        ],
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100vh", backgroundColor: "#282c34" }}>
            <div style={{ width: "80%", backgroundColor: "#20232a", padding: "20px", display: "flex", flexDirection: "column", alignItems: "center" }}>
                <h1 style={{ color: "#61dafb" }}>Stock Transactions</h1>
                <select onChange={(e) => setSelectedCompany(e.target.value)} style={{ padding: "10px", fontSize: "16px", marginBottom: "10px" }}>
                    <option value="">Select a Company</option>
                    {companies.map((company, index) => <option key={index} value={company}>{company}</option>)}
                </select>
                <button onClick={handleFetchTransactions} style={{ padding: "10px", fontSize: "16px", marginBottom: "10px" }}>View Transactions</button>
                {labels.length > 0 ? <Line data={chartData} /> : <p style={{ color: "white" }}>No data available.</p>}
            </div>
        </div>
    );
};

export default App;
