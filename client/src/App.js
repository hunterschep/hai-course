import { useState, useEffect, useRef } from 'react';

const url = process.env.NODE_ENV === 'production' 
  ? 'https://course-tools-demo.onrender.com/' 
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
    <div className="min-h-screen bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center p-4">
      <div className="bg-white w-full max-w-3xl rounded-xl shadow-lg p-6">
        <h1 className="text-3xl text-center font-semibold mb-6 text-gray-800">Chat with Simple Bot</h1>
        <div className="overflow-y-auto h-80 mb-5 p-4 border border-gray-200 rounded-lg bg-gray-50">
          {chatHistory.length === 0 ? (
            <p className="text-gray-500 text-center">No messages yet. Start the conversation!</p>
          ) : (
            chatHistory.map((chat, index) => (
              <div key={index} className={`flex ${chat.sender === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
                <div className={`flex items-center ${chat.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300 text-black'} rounded-xl p-4 max-w-xs shadow w-full break-words`}>
                  <div className="mr-3 flex-shrink-0">
                    <img 
                      src="/user.png" 
                      alt="User" 
                      className="w-10 h-10 rounded-full object-cover"
                    />
                  </div>
                  <div className="text-base w-full break-words">
                    {chat.message}
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>
        <div className="flex mt-4">
          <input 
            type="text" 
            placeholder="Type your message here..." 
            value={message} 
            onChange={handleMessage} 
            onKeyPress={handleKeyPress} 
            className="flex-grow p-4 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button 
            className="bg-blue-500 text-white px-6 py-4 rounded-r-lg hover:bg-blue-600 transition duration-200" 
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
