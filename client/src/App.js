import { useState, useEffect, useRef } from 'react';

const url = process.env.NODE_ENV === 'production' 
  ? 'https://hai-fastapi.vercel.app/' 
  : 'http://127.0.0.1:8000/';

function App() {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const chatEndRef = useRef(null);

  const sendMessage = async () => {
    if (!message.trim()) return;

    const newChatHistory = [...chatHistory, { sender: 'user', message }];
    setChatHistory(newChatHistory);

    try {
      const res = await fetch(`${url}query`, {
        method: 'POST',
        body: JSON.stringify({ prompt: message }),
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const data = await res.json();
      setChatHistory([...newChatHistory, { sender: 'bot', message: data.response }]);
    } catch (error) {
      console.error("Error fetching the response:", error);
      setChatHistory([...newChatHistory, { sender: 'bot', message: "An error occurred. Please try again." }]);
    }

    setMessage("");
  };

  const handleMessage = (e) => {
    setMessage(e.target.value);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  return (
    <div className="min-h-screen bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-center p-4">
      <div className="bg-white w-full max-w-2xl rounded-lg shadow-lg p-6 space-y-6">
        <h1 className="text-3xl text-center font-bold text-gray-900">Chat with Simple Bot</h1>
        <div className="overflow-y-auto h-80 mb-5 p-4 border border-gray-200 rounded-lg bg-gray-50 shadow-inner">
          {chatHistory.length === 0 ? (
            <p className="text-gray-500 text-center">No messages yet. Start the conversation!</p>
          ) : (
            chatHistory.map((chat, index) => (
              <div key={index} className={`flex ${chat.sender === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
                {chat.sender === 'bot' && (
                  <img 
                    src="/user.png" 
                    alt="Bot" 
                    className="w-8 h-8 rounded-full object-cover mr-3"
                  />
                )}
                <div className={`flex items-center ${chat.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-black'} rounded-lg p-3 max-w-fit shadow w-auto`}>
                  <div className="text-base break-words">
                    {chat.message}
                  </div>
                </div>
                {chat.sender === 'user' && (
                  <img 
                    src="/user.png" 
                    alt="User" 
                    className="w-8 h-8 rounded-full object-cover ml-3"
                  />
                )}
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>
        <div className="flex">
          <input 
            type="text" 
            placeholder="Type your message..." 
            value={message} 
            onChange={handleMessage} 
            onKeyPress={handleKeyPress} 
            className="flex-grow p-4 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200"
          />
          <button 
            className="bg-indigo-600 text-white px-6 py-4 rounded-r-lg hover:bg-indigo-700 transition duration-200" 
            onClick={sendMessage}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
