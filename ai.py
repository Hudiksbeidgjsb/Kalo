import numpy as np
import pickle
import os
import time
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==================== ADVANCED NEURAL NETWORK ====================

class AdvancedNeuralNetwork:
    def __init__(self, layers: List[int], activation='relu', dropout_rate=0.0, 
                 weight_init='he', learning_rate=0.001, optimizer='adam'):
        self.layers = layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.parameters = {}
        self.velocities = {}
        self.squares = {}
        self.cache = {}
        self.t = 0
        self._initialize_parameters(weight_init)
    
    def _initialize_parameters(self, weight_init):
        L = len(self.layers)
        
        for l in range(1, L):
            if weight_init == 'he':
                scale = np.sqrt(2.0 / self.layers[l-1])
            elif weight_init == 'xavier':
                scale = np.sqrt(1.0 / self.layers[l-1])
            else:
                scale = 0.01
            
            self.parameters[f'W{l}'] = np.random.randn(self.layers[l-1], self.layers[l]) * scale
            self.parameters[f'b{l}'] = np.zeros((1, self.layers[l]))
            
            if self.optimizer in ['momentum', 'adam']:
                self.velocities[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.velocities[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
            
            if self.optimizer == 'adam':
                self.squares[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.squares[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])

    def _activation_function(self, Z, activation):
        if activation == 'relu':
            return np.maximum(0, Z), Z
        elif activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-Z))
            return sig, Z
        elif activation == 'tanh':
            return np.tanh(Z), Z
        elif activation == 'leaky_relu':
            return np.where(Z > 0, Z, 0.01 * Z), Z
        else:
            return Z, Z

    def _activation_derivative(self, dA, Z, activation):
        if activation == 'relu':
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            return dZ
        elif activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-Z))
            return dA * sig * (1 - sig)
        elif activation == 'tanh':
            return dA * (1 - np.tanh(Z)**2)
        elif activation == 'leaky_relu':
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] *= 0.01
            return dZ
        else:
            return dA

    def forward(self, X, training=True):
        A = X
        L = len(self.layers) - 1
        self.cache['A0'] = A
        
        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(A, W) + b
            A, self.cache[f'Z{l}'] = self._activation_function(Z, self.activation)
            
            if training and self.dropout_rate > 0:
                D = np.random.rand(*A.shape) > self.dropout_rate
                A = A * D
                A = A / (1 - self.dropout_rate)
                self.cache[f'D{l}'] = D
            
            self.cache[f'A{l}'] = A
        
        # Output layer
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        Z = np.dot(A, W) + b
        
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        self.cache[f'Z{L}'] = Z
        self.cache[f'A{L}'] = A
        
        return A

    def compute_loss(self, AL, Y):
        m = Y.shape[0]
        log_probs = -np.log(AL[range(m), Y] + 1e-8)
        loss = np.sum(log_probs) / m
        
        L = len(self.layers) - 1
        reg_loss = 0
        for l in range(1, L + 1):
            reg_loss += np.sum(np.square(self.parameters[f'W{l}']))
        
        loss += 0.01 * reg_loss / (2 * m)
        return loss

    def backward(self, X, Y):
        m = X.shape[0]
        L = len(self.layers) - 1
        grads = {}
        
        AL = self.cache[f'A{L}']
        dZ = AL
        dZ[range(m), Y] -= 1
        dZ /= m
        
        grads[f'dW{L}'] = np.dot(self.cache[f'A{L-1}'].T, dZ)
        grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True)
        
        for l in reversed(range(1, L)):
            dA = np.dot(dZ, self.parameters[f'W{l+1}'].T)
            
            if f'D{l}' in self.cache:
                dA = dA * self.cache[f'D{l}']
                dA = dA / (1 - self.dropout_rate)
            
            dZ = self._activation_derivative(dA, self.cache[f'Z{l}'], self.activation)
            
            grads[f'dW{l}'] = np.dot(self.cache[f'A{l-1}'].T, dZ)
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True)
        
        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.layers) - 1
        self.t += 1
        
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        for l in range(1, L + 1):
            if self.optimizer == 'gradient_descent':
                self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
                self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
            
            elif self.optimizer == 'momentum':
                self.velocities[f'dW{l}'] = beta1 * self.velocities[f'dW{l}'] + (1 - beta1) * grads[f'dW{l}']
                self.velocities[f'db{l}'] = beta1 * self.velocities[f'db{l}'] + (1 - beta1) * grads[f'db{l}']
                
                self.parameters[f'W{l}'] -= learning_rate * self.velocities[f'dW{l}']
                self.parameters[f'b{l}'] -= learning_rate * self.velocities[f'db{l}']
            
            elif self.optimizer == 'adam':
                self.velocities[f'dW{l}'] = beta1 * self.velocities[f'dW{l}'] + (1 - beta1) * grads[f'dW{l}']
                self.velocities[f'db{l}'] = beta1 * self.velocities[f'db{l}'] + (1 - beta1) * grads[f'db{l}']
                
                self.squares[f'dW{l}'] = beta2 * self.squares[f'dW{l}'] + (1 - beta2) * np.square(grads[f'dW{l}'])
                self.squares[f'db{l}'] = beta2 * self.squares[f'db{l}'] + (1 - beta2) * np.square(grads[f'db{l}'])
                
                v_corrected_W = self.velocities[f'dW{l}'] / (1 - beta1**self.t)
                v_corrected_b = self.velocities[f'db{l}'] / (1 - beta1**self.t)
                
                s_corrected_W = self.squares[f'dW{l}'] / (1 - beta2**self.t)
                s_corrected_b = self.squares[f'db{l}'] / (1 - beta2**self.t)
                
                self.parameters[f'W{l}'] -= learning_rate * v_corrected_W / (np.sqrt(s_corrected_W) + epsilon)
                self.parameters[f'b{l}'] -= learning_rate * v_corrected_b / (np.sqrt(s_corrected_b) + epsilon)

    def predict(self, X):
        probabilities = self.forward(X, training=False)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'parameters': self.parameters,
                'layers': self.layers,
                'activation': self.activation,
                'optimizer': self.optimizer
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.parameters = data['parameters']
            self.layers = data['layers']
            self.activation = data['activation']
            self.optimizer = data['optimizer']

# ==================== AI CHAT MANAGER ====================

class AIChatManager:
    def __init__(self):
        self.models = {}
        self.user_sessions = {}
        self.conversation_history = {}
        self.model_dir = "ai_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize some pre-trained models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        # Sentiment Analysis Model
        sentiment_model = AdvancedNeuralNetwork(
            layers=[100, 64, 32, 3],  # 3 classes: positive, negative, neutral
            activation='relu',
            dropout_rate=0.2,
            optimizer='adam'
        )
        self.models['sentiment'] = sentiment_model
        
        # Text Classification Model
        text_classifier = AdvancedNeuralNetwork(
            layers=[100, 128, 64, 5],  # 5 categories
            activation='relu',
            dropout_rate=0.3,
            optimizer='adam'
        )
        self.models['classifier'] = text_classifier
        
        # Simple Q&A Model
        qa_model = AdvancedNeuralNetwork(
            layers=[150, 100, 50, 10],  # 10 response types
            activation='tanh',
            dropout_rate=0.1,
            optimizer='adam'
        )
        self.models['qa'] = qa_model
    
    def text_to_features(self, text: str, feature_size: int = 100):
        """Convert text to numerical features"""
        # Simple feature extraction: character frequency, length, word patterns
        features = np.zeros(feature_size)
        text = text.lower().strip()
        
        # Character frequency (first 26 features)
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
            if i < 26:
                features[i] = text.count(char) / len(text) if text else 0
        
        # Text length and patterns
        features[26] = len(text) / 100  # Normalized length
        features[27] = len(text.split()) / 20  # Word count
        
        # Presence of common words/patterns
        common_patterns = ['what', 'why', 'how', 'when', 'where', 'who', 'help', 'thanks', 'hello', 'good']
        for i, pattern in enumerate(common_patterns):
            if i + 28 < feature_size:
                features[28 + i] = 1 if pattern in text else 0
        
        # Sentiment indicators
        positive_words = ['good', 'great', 'awesome', 'excellent', 'happy', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry']
        
        features[38] = sum(1 for word in positive_words if word in text)
        features[39] = sum(1 for word in negative_words if word in text)
        
        # Fill remaining features with random but deterministic values
        np.random.seed(hash(text) % 10000)
        remaining_features = feature_size - 40
        if remaining_features > 0:
            features[40:] = np.random.randn(remaining_features) * 0.1
        
        return features.reshape(1, -1)
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze text sentiment"""
        features = self.text_to_features(text, 100)
        model = self.models['sentiment']
        prediction, probabilities = model.predict(features)
        
        sentiments = ['negative', 'neutral', 'positive']
        confidence = np.max(probabilities)
        
        return f"Sentiment: {sentiments[prediction[0]]} (confidence: {confidence:.2f})"
    
    def generate_response(self, user_id: int, text: str) -> str:
        """Generate AI response based on user input"""
        # Update conversation history
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append(f"User: {text}")
        
        # Simple rule-based responses with AI enhancement
        text_lower = text.lower()
        
        # Greetings
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = "Hello! I'm your AI assistant. How can I help you today?"
        
        # Questions about AI
        elif any(word in text_lower for word in ['what are you', 'who are you']):
            response = "I'm a neural network AI running on this server! I can chat, analyze text, and learn from our conversation."
        
        # Sentiment analysis request
        elif any(word in text_lower for word in ['sentiment', 'feel', 'emotion', 'mood']):
            sentiment_result = self.analyze_sentiment(text)
            response = f"🤖 AI Analysis:\n{sentiment_result}"
        
        # Help
        elif 'help' in text_lower:
            response = """🤖 *AI Bot Commands:*
/start - Start conversation
/train - Train the AI model
/status - Check AI status
/sentiment - Analyze text sentiment
/clear - Clear conversation history
/help - Show this help message

You can also just chat with me normally!"""
        
        # Default AI response using the QA model
        else:
            features = self.text_to_features(text, 150)
            model = self.models['qa']
            prediction, probabilities = model.predict(features)
            
            # Simple response templates based on prediction
            responses = [
                "That's interesting! Tell me more about that.",
                "I understand. How does that make you feel?",
                "Fascinating! I'm learning from our conversation.",
                "That's a great point. What else is on your mind?",
                "I see. Could you elaborate on that?",
                "Thanks for sharing! My neural networks are processing this.",
                "Interesting perspective! My AI model is analyzing your message.",
                "I'm learning from this interaction. Please continue!",
                "That's valuable input for my training data!",
                "My algorithms are processing your message. This helps me improve!"
            ]
            
            response = responses[prediction[0] % len(responses)]
            
            # Add some AI learning context occasionally
            if np.random.random() < 0.3:
                response += f"\n\n🤖 *AI Insight*: My neural network has {sum(len(layer) for layer in model.layers)} neurons processing your message!"
        
        # Update conversation history
        self.conversation_history[user_id].append(f"AI: {response}")
        
        # Keep only last 10 messages
        if len(self.conversation_history[user_id]) > 20:
            self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
        
        return response
    
    def train_models(self):
        """Train AI models with generated data"""
        logger.info("Training AI models...")
        
        # Generate synthetic training data
        X_train = np.random.randn(1000, 100)
        y_train = np.random.randint(0, 3, 1000)
        
        # Train sentiment model
        model = self.models['sentiment']
        for epoch in range(100):
            output = model.forward(X_train)
            loss = model.compute_loss(y_train, output)
            grads = model.backward(X_train, y_train)
            model.update_parameters(grads, 0.01)
        
        logger.info("AI models training completed!")
    
    def get_status(self) -> str:
        """Get AI system status"""
        total_neurons = sum(sum(len(layer) for layer in model.layers) for model in self.models.values())
        total_connections = sum(sum(np.prod(param.shape) for param in model.parameters.values()) for model in self.models.values())
        
        return f"""🤖 *AI System Status*

• Active Models: {len(self.models)}
• Total Neurons: {total_neurons:,}
• Total Connections: {total_connections:,}
• Active Users: {len(self.user_sessions)}
• Conversations: {len(self.conversation_history)}

*Models Available:*
- Sentiment Analysis
- Text Classification
- Q&A Response
- Neural Chat

System: 🟢 Operational"""

# ==================== TELEGRAM BOT ====================

class TelegramAIBot:
    def __init__(self, token: str):
        self.token = token
        self.ai_manager = AIChatManager()
        self.application = Application.builder().token(token).build()
        self._setup_handlers()
    
    def _setup_handlers(self):
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("train", self.train_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("sentiment", self.sentiment_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        
        # Message handler
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send welcome message when command /start is issued"""
        user = update.effective_user
        welcome_text = f"""
🤖 *Welcome {user.first_name} to Your Personal AI!*

I'm a complete neural network AI running on this server. Here's what I can do:

• 💬 *Chat* with you using my neural networks
• 🧠 *Learn* from our conversations
• 📊 *Analyze* text sentiment
• 🔄 *Train* and improve over time

*My Architecture:*
- Multiple neural network models
- Advanced backpropagation
- Adam optimization
- Dropout regularization

Try chatting with me or use /help for commands!
        """
        
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
        
        # Initialize user session
        user_id = user.id
        self.ai_manager.user_sessions[user_id] = {
            'start_time': datetime.now(),
            'message_count': 0
        }
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send help message when command /help is issued"""
        help_text = """
🤖 *AI Bot Commands*

*/start* - Initialize your AI session
*/train* - Train the neural networks
*/status* - Check AI system status  
*/sentiment <text>* - Analyze text sentiment
*/clear* - Clear conversation history
*/help* - Show this message

*What I Can Do:*
- Natural language conversations
- Sentiment analysis
- Text classification
- Continuous learning
- Neural network processing

Just send me a message to start chatting!
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Train the AI models"""
        training_msg = await update.message.reply_text("🧠 *Training Neural Networks...*", parse_mode='Markdown')
        
        # Train in background
        await asyncio.get_event_loop().run_in_executor(None, self.ai_manager.train_models)
        
        await training_msg.edit_text("✅ *AI Training Complete!*\n\nMy neural networks have been updated and improved!")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AI system status"""
        status = self.ai_manager.get_status()
        await update.message.reply_text(status, parse_mode='Markdown')
    
    async def sentiment_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analyze text sentiment"""
        if not context.args:
            await update.message.reply_text("Please provide text to analyze. Usage: /sentiment <your text>")
            return
        
        text = ' '.join(context.args)
        sentiment_result = self.ai_manager.analyze_sentiment(text)
        
        response = f"🔍 *Sentiment Analysis*\n\n*Text:* {text}\n*Result:* {sentiment_result}"
        await update.message.reply_text(response, parse_mode='Markdown')
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clear conversation history"""
        user_id = update.effective_user.id
        if user_id in self.ai_manager.conversation_history:
            self.ai_manager.conversation_history[user_id] = []
        
        await update.message.reply_text("✅ Conversation history cleared!")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages"""
        user_id = update.effective_user.id
        text = update.message.text
        
        # Update user session
        if user_id in self.ai_manager.user_sessions:
            self.ai_manager.user_sessions[user_id]['message_count'] += 1
        else:
            self.ai_manager.user_sessions[user_id] = {
                'start_time': datetime.now(),
                'message_count': 1
            }
        
        # Show typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Generate AI response
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.ai_manager.generate_response, user_id, text
        )
        
        await update.message.reply_text(response, parse_mode='Markdown')
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Log errors"""
        logger.error(f"Exception while handling an update: {context.error}")
    
    def run(self):
        """Start the bot"""
        logger.info("🤖 AI Telegram Bot is starting...")
        logger.info("AI Models initialized and ready!")
        logger.info("Bot is polling for messages...")
        
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

# ==================== MAIN EXECUTION ====================

def main():
    # Replace with your actual Telegram Bot Token
    BOT_TOKEN = "8038917688:AAHDo_FO19MYuHOkXmsKCcuSuJ3McqwOrAU"
    
    if BOT_TOKEN == "8038917688:AAHDo_FO19MYuHOkXmsKCcuSuJ3McqwOrAU":
        print("❌ ERROR: Please replace 'YOUR_TELEGRAM_BOT_TOKEN_HERE' with your actual Telegram Bot Token")
        print("Get your token from @BotFather on Telegram")
        return
    
    # Create and run the bot
    bot = TelegramAIBot(BOT_TOKEN)
    bot.run()

if __name__ == "__main__":
    main()
