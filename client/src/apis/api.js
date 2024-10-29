// /src/api/api.js

const API_URL = process.env.NODE_ENV === 'production' 
  ? 'https://hai-course.onrender.com' 
  : 'http://127.0.0.1:8000/';

export const sendMessageToAPI = async (message, data, chatHistory, setChatHistory, setMessage, setIsThinking) => {
  if (!message.trim() || !data) return;

  const newChatHistory = [...chatHistory, { sender: 'user', message }];
  setChatHistory(newChatHistory);
  setMessage("");
  setIsThinking(true);

  try {
    const apiUrl = new URL('/query/', API_URL).toString();
    const res = await fetch(apiUrl, {
      method: 'POST',
      body: JSON.stringify({ prompt: message, data }),
      headers: { 'Content-Type': 'application/json' }
    });

    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

    const { chartSpec, description, table } = await res.json();
    setChatHistory(prev => [...prev, { sender: 'bot', message: description, chartSpec, table}]);
  } catch (error) {
    console.error("Error fetching the response:", error);
    setChatHistory(prev => [...prev, { sender: 'bot', message: "An error occurred. Please try again." }]);
  } finally {
    setIsThinking(false);
  }
};
