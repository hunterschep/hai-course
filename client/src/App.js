// /src/app.js
import { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3-dsv";
import { VegaLite } from "react-vega";
import { sendMessageToAPI } from "./apis/api";

function App() {
    const [message, setMessage] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [data, setData] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isThinking, setIsThinking] = useState(false);
    const [showPreview, setShowPreview] = useState(true);
    const chatEndRef = useRef(null);

    // Rename sendMessageToAPI to sendMessage in the callback
    const sendMessage = useCallback(() => {
        if (data == null) {
            setChatHistory((prev) => [
                ...prev,
                { sender: "bot", message: "Please upload a CSV file first." },
            ]);
            // clear the message currently in the send message box
            setMessage("");
            return;
        }

        sendMessageToAPI(
            message,
            data,
            chatHistory,
            setChatHistory,
            setMessage,
            setIsThinking
        );
        setMessage("");
    }, [message, data, chatHistory]);

    const handleFileUpload = useCallback((file) => {
        if (file?.type === "text/csv") {
            const reader = new FileReader();
            reader.onload = (event) => {
                const csvData = d3.csvParse(event.target.result, d3.autoType);
                setData(csvData);
                setChatHistory((prev) => [
                    ...prev,
                    {
                        sender: "bot",
                        message: "Dataset uploaded successfully.",
                    },
                ]);
            };
            reader.readAsText(file);
        } else {
            setChatHistory((prev) => [
                ...prev,
                { sender: "bot", message: "Please upload a valid CSV file." },
            ]);
        }
    }, []);

    const handleDrop = useCallback(
        (e) => {
            e.preventDefault();
            setIsDragging(false);
            handleFileUpload(e.dataTransfer.files[0]);
        },
        [handleFileUpload]
    );

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatHistory]);

    const VegaLiteWithErrorHandling = ({ spec }) => {
        try {
            return <VegaLite spec={spec} width={450} height={250} />;
        } catch (error) {
            console.error("Error rendering VegaLite chart:", error);
            return (
                <p className="text-red-500">
                    Unable to render chart. Invalid specification.
                </p>
            );
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-r from-purple-500 via-pink-500 to-red-400 flex items-center justify-center p-6">
            <div className="bg-white w-full max-w-5xl rounded-xl shadow-2xl p-10 space-y-10">
                <h1 className="text-4xl text-center font-extrabold text-gray-900 tracking-tight">
                    Hunter's Data Visualization Bot
                </h1>
                <p className="text-center text-lg text-gray-600">
                    Upload your dataset and ask questions to generate
                    interactive visualizations. For example, try asking "Show me
                    mpg by origin".
                </p>

                {/* File Upload */}
                <div
                    className={`border-4 ${
                        isDragging ? "border-indigo-500" : "border-gray-300"
                    } border-dashed rounded-lg p-10 text-center cursor-pointer hover:bg-gray-50 transition duration-300 ease-in-out`}
                    onDragOver={(e) => {
                        e.preventDefault();
                        setIsDragging(true);
                    }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById("fileInput").click()}
                >
                    <p className="text-lg text-gray-500">
                        Drag & Drop a CSV file here, or click to upload
                    </p>
                    <input
                        type="file"
                        id="fileInput"
                        accept=".csv"
                        onChange={(e) => handleFileUpload(e.target.files[0])}
                        className="hidden"
                    />
                </div>

                {/* Data Preview */}
                {data && (
                    <div className="mt-6">
                        {/* Preview table, only show if showPreview is true */}
                        {showPreview && (
                            <div className="mt-4 overflow-auto border border-gray-300 rounded-lg p-6 bg-gray-50 shadow-inner max-h-60">
                                <table className="table-auto w-full text-left text-sm">
                                    <thead>
                                        <tr>
                                            {Object.keys(data[0]).map((col) => (
                                                <th
                                                    key={col}
                                                    className="px-6 py-3 border-b border-gray-200"
                                                >
                                                    {col}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {data
                                            .slice(0, 10)
                                            .map((row, rowIndex) => (
                                                <tr key={rowIndex}>
                                                    {Object.values(row).map(
                                                        (value, colIndex) => (
                                                            <td
                                                                key={colIndex}
                                                                className="px-6 py-3 border-b border-gray-200"
                                                            >
                                                                {value}
                                                            </td>
                                                        )
                                                    )}
                                                </tr>
                                            ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                        {/* Center the button */}
                        <div className="flex justify-center pt-4">
                            <button
                                onClick={() => setShowPreview(!showPreview)}
                                className="bg-indigo-600 text-white px-6 py-3 rounded-full shadow-lg hover:bg-indigo-700 transition duration-300 ease-in-out"
                            >
                                {showPreview ? "Hide Preview" : "Show Preview"}
                            </button>
                        </div>
                    </div>
                )}

                {/* Chat History */}
                <div className="overflow-y-auto h-96 mb-6 p-6 border border-gray-300 rounded-lg bg-white shadow-lg">
                    {chatHistory.length === 0 ? (
                        <p className="text-gray-500 text-center">
                            No messages yet. Start the conversation!
                        </p>
                    ) : (
                        chatHistory.map((chat, index) => (
                            <div
                                key={index}
                                className={`mb-5 flex ${
                                    chat.sender === "user"
                                        ? "justify-end"
                                        : "justify-start"
                                } items-center`}
                            >
                                {/* Show bot image when it's the bot sending the message */}
                                {chat.sender === "bot" && (
                                    <img
                                        src={`${process.env.PUBLIC_URL}/user.png`}
                                        alt="Bot"
                                        className="w-10 h-10 rounded-full object-cover mr-3"
                                    />
                                )}
                                {/* Chat Bubble */}
                                <div
                                    className={`flex flex-col items-${
                                        chat.sender === "user" ? "end" : "start"
                                    } max-w-lg mb-5`} // Adjusted the overall container
                                >
                                    {/* Render the VegaLite chart if chat.chartSpec exists */}
                                    {chat.chartSpec && (
                                        <div className="mb-3 max-w-full"> {/* Full width container for the chart */}
                                            <VegaLiteWithErrorHandling
                                                spec={chat.chartSpec}
                                            />
                                        </div>
                                    )}

                                    {/* Chat bubble containing the message text only */}
                                    <div
                                        className={`p-4 rounded-lg shadow-md ${
                                            chat.sender === "user"
                                                ? "bg-blue-500 text-white"
                                                : "bg-gray-200 text-gray-900"
                                        }`}
                                    >
                                        {/* Message text */}
                                        <div className="text-base break-words">
                                            {chat.message}
                                        </div>
                                    </div>
                                </div>



                                {/* Show user image when it's the user sending the message */}
                                {chat.sender === "user" && (
                                    <img
                                        src={`${process.env.PUBLIC_URL}/user.png`}
                                        alt="User"
                                        className="w-10 h-10 rounded-full object-cover ml-3"
                                    />
                                )}
                            </div>
                        ))
                    )}

                    {isThinking && (
                        <div className="text-gray-500 text-center mb-5">
                            Thinking...
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </div>

                {/* Message Input */}
                <div className="flex space-x-1">
                    <input
                        type="text"
                        placeholder="Type your message..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                        className="flex-grow p-4 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-indigo-600 transition duration-300"
                    />
                    <div className="flex space-x-1">
                        <button
                            className="bg-indigo-600 text-white px-8 py-4 rounded-lg hover:bg-indigo-700 transition duration-300"
                            onClick={sendMessage}
                        >
                            Send
                        </button>
                        <button
                            className="bg-indigo-600 text-white px-8 py-4 rounded-lg hover:bg-indigo-700 transition duration-300"
                            onClick={setChatHistory.bind(null, [])}
                        >
                            Clear
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
