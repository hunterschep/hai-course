// /src/app.js
import { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3-dsv";
import { VegaLite } from "react-vega";
import { sendMessageToAPI } from "./apis/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import React from "react";

const svgPath = process.env.PUBLIC_URL + "/fade-stagger-circles.svg";

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
        const timeout = setTimeout(() => {
            chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
        }, 100); // slight delay to allow DOM updates
        return () => clearTimeout(timeout); // clean up on unmount
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
        <div className="min-h-screen bg-gradient-to-r from-purple-500 via-pink-500 to-red-400 flex items-center justify-center p-4 sm:p-6">
            <div className="bg-white w-full max-w-lg sm:max-w-2xl lg:max-w-5xl rounded-xl shadow-2xl p-6 sm:p-10 space-y-6 sm:space-y-10">
                <h1 className="text-2xl sm:text-4xl text-center font-extrabold text-gray-900 tracking-tight">
                    Hunter's Data Visualization Bot
                </h1>
                <p className="text-center text-sm sm:text-lg text-gray-600">
                    Upload your dataset and ask questions to generate interactive visualizations and analysis!
                </p>
    
                {/* File Upload */}
                <div
                    className={`border-4 ${isDragging ? "border-indigo-500" : "border-gray-300"} border-dashed rounded-lg p-6 sm:p-10 text-center cursor-pointer hover:bg-gray-50 transition duration-300 ease-in-out`}
                    onDragOver={(e) => {
                        e.preventDefault();
                        setIsDragging(true);
                    }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById("fileInput").click()}
                >
                    <p className="text-sm sm:text-lg text-gray-500">
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
                    <div className="mt-4 sm:mt-6">
                        {showPreview && (
                            <div className="mt-4 overflow-auto border border-gray-300 rounded-lg p-4 sm:p-6 bg-gray-50 shadow-inner max-h-40 sm:max-h-60">
                                <table className="table-auto w-full text-left text-xs sm:text-sm">
                                    <thead>
                                        <tr>
                                            {Object.keys(data[0]).map((col) => (
                                                <th key={col} className="px-4 sm:px-6 py-2 sm:py-3 border-b border-gray-200">
                                                    {col}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {data.slice(0, 10).map((row, rowIndex) => (
                                            <tr key={rowIndex}>
                                                {Object.values(row).map((value, colIndex) => (
                                                    <td key={colIndex} className="px-4 sm:px-6 py-2 sm:py-3 border-b border-gray-200">
                                                        {value}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                        <div className="flex justify-center pt-4">
                            <button
                                onClick={() => setShowPreview(!showPreview)}
                                className="bg-indigo-600 text-white px-4 sm:px-6 py-2 sm:py-3 rounded-full shadow-lg hover:bg-indigo-700 transition duration-300 ease-in-out"
                            >
                                {showPreview ? "Hide Preview" : "Show Preview"}
                            </button>
                        </div>
                    </div>
                )}
    
                {/* Chat History */}
                <div className="overflow-y-auto h-48 sm:h-96 mb-6 p-4 sm:p-6 border border-gray-300 rounded-lg bg-white shadow-lg">
                    {chatHistory.length === 0 ? (
                        <p className="text-gray-500 text-center">
                            What can I help with?
                        </p>
                    ) : (
                        chatHistory.map((chat, index) => (
                            <div key={index} className={`mb-4 flex ${chat.sender === "user" ? "justify-end" : "justify-start"} items-center`}>
                              {chat.sender === "bot" && (
                                <img src={`${process.env.PUBLIC_URL}/user.png`} alt="Bot" className="w-8 sm:w-10 h-8 sm:h-10 rounded-full object-cover mr-3" />
                              )}
                              <div className={`flex flex-col items-${chat.sender === "user" ? "end" : "start"} max-w-xs sm:max-w-lg`}>
                                {chat.chartSpec && (
                                  <div className="mb-2 sm:mb-3 max-w-full">
                                    <VegaLiteWithErrorHandling spec={chat.chartSpec} />
                                  </div>
                                )}
                                {chat.table && (
                                  <div className="mb-2 sm:mb-3 max-w-full bg-white p-4 rounded-lg shadow-lg">
                                    {/* Render Markdown tables */}
                                    <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        children={chat.table}
                                        className="markdown-table text-sm sm:text-base break-words"
                                    />
                                  </div>
                                )}
                                <div className={`p-3 sm:p-4 rounded-lg shadow-md ${chat.sender === "user" ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-900"}`}>
                                  <div className="text-xs sm:text-base break-words">{chat.message}</div>
                                </div>
                              </div>
                              {chat.sender === "user" && (
                                <img src={`${process.env.PUBLIC_URL}/user.png`} alt="User" className="w-8 sm:w-10 h-8 sm:h-10 rounded-full object-cover ml-3" />
                              )}
                            </div>
                          ))
                    )}
                    {isThinking && (
                        <div className="text-gray-500 text-center mb-5 flex items-center justify-center">
                            <img src={svgPath} alt="Loading..." className="h-5 w-5 mr-3 animate-spin" />
                            Thinking
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </div>
    
                {/* Message Input */}
                <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-1">
                    <input
                        type="text"
                        placeholder="Message data visualization bot..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                        className="flex-grow p-3 sm:p-4 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-600 transition duration-300"
                    />
                    <button
                        className="bg-indigo-600 text-white px-4 sm:px-8 py-3 sm:py-4 rounded-lg hover:bg-indigo-700 transition duration-300"
                        onClick={sendMessage}
                    >
                        Send
                    </button>
                    <button
                        className="bg-indigo-600 text-white px-4 sm:px-8 py-3 sm:py-4 rounded-lg hover:bg-indigo-700 transition duration-300"
                        onClick={() => setChatHistory([])}
                    >
                        Clear Messages
                    </button>
                </div>
            </div>
        </div>
    );}
    

export default App;