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

    // Add user's message to chat history
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

      // Add bot's response to chat history
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

  // Scroll to the bottom whenever chatHistory changes
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-5">
      <div className="bg-white w-full max-w-2xl rounded-lg shadow-lg p-6">
        <h1 className="text-4xl text-center font-bold mb-8">Chat with Simple Bot</h1>
        <div className="overflow-y-auto max-h-80 mb-5 p-3 border rounded-lg bg-gray-50">
          {chatHistory.length === 0 ? (
            <p className="text-gray-500 text-center">No messages yet. Start the conversation!</p>
          ) : (
            chatHistory.map((chat, index) => (
              <div key={index} className={`flex ${chat.sender === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
                <div className={`flex items-center ${chat.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300 text-black'} rounded-lg p-3 max-w-xs`}>
                  <div className="mr-2">
                    <img 
                      src={chat.sender === 'user' ? '/user.png' : '/bot.png'} 
                      alt={chat.sender === 'user' ? 'User' : 'Bot'} 
                      className="w-8 h-8 rounded-full"
                    />
                  </div>
                  <div className="text-sm">
                    {chat.message}
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>
        <div className="flex">
          <input 
            type="text" 
            placeholder="Type your message here..." 
            value={message} 
            onChange={handleMessage} 
            onKeyPress={handleKeyPress} 
            className="flex-grow p-3 border border-gray-300 rounded-l-lg focus:outline-none"
          />
          <button 
            className="bg-blue-500 text-white px-4 py-3 rounded-r-lg hover:bg-blue-600 transition duration-200" 
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
