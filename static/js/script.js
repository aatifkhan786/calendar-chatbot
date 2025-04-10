// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    const messagesDiv = document.getElementById('messages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const typingIndicator = messagesDiv.querySelector('.typing-indicator');
    const initialPlaceholder = document.getElementById('initial-placeholder'); // Get placeholder
    let placeholderVisible = true; // Flag to track placeholder state

    // --- Hide Placeholder ---
    function hidePlaceholder() {
        if (placeholderVisible && initialPlaceholder) {
            initialPlaceholder.style.display = 'none';
            placeholderVisible = false;
        }
    }

    // --- Textarea Auto-Resize ---
    const initialHeight = userInput.scrollHeight;
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto'; // Reset height
        let newHeight = userInput.scrollHeight;
        const maxHeight = 200;
        if (newHeight > maxHeight) {
            newHeight = maxHeight;
            userInput.style.overflowY = 'auto';
        } else {
            userInput.style.overflowY = 'hidden';
        }
        userInput.style.height = `${newHeight}px`;
        sendButton.disabled = userInput.value.trim().length === 0;
    });

    // --- Typing Indicator Functions ---
    function showTypingIndicator() {
        // Ensure placeholder is hidden before showing typing
        hidePlaceholder();
        if (typingIndicator) {
            typingIndicator.style.display = 'flex';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    }
    function hideTypingIndicator() {
        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }
    }

    // --- Add Message Function ---
    function addMessage(text, sender) {
        // *** Hide the placeholder when the first message is added ***
        hidePlaceholder();

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        const iconContainer = document.createElement('div');
        iconContainer.classList.add('message-icon');
        const icon = document.createElement('i');
        if (sender === 'bot') {
            iconContainer.classList.add('bot-icon-container');
            icon.classList.add('fas', 'fa-robot');
        } else {
            iconContainer.classList.add('user-icon-container');
            icon.classList.add('fas', 'fa-user');
        }
        iconContainer.appendChild(icon);
        messageDiv.appendChild(iconContainer);

        const contentContainer = document.createElement('div');
        contentContainer.classList.add('message-content');
        const textSpan = document.createElement('span');
        if (sender === 'user') {
            const sanitizedText = text.replace(/</g, "<").replace(/>/g, ">");
            textSpan.innerHTML = sanitizedText;
        } else {
            textSpan.innerHTML = text;
        }
        contentContainer.appendChild(textSpan);
        messageDiv.appendChild(contentContainer);

        if (typingIndicator) {
            messagesDiv.insertBefore(messageDiv, typingIndicator);
        } else {
            messagesDiv.appendChild(messageDiv);
        }

        messagesDiv.scrollTo({ top: messagesDiv.scrollHeight, behavior: 'smooth' });
    }

    // --- Send Message Function ---
    async function sendMessage(messageText = userInput.value.trim()) { // Allow passing message text
        if (!messageText) return;

        // If the message came from userInput, clear it
        if (messageText === userInput.value.trim()) {
             addMessage(messageText, 'user'); // Add user message bubble
             userInput.value = '';
             userInput.style.height = 'auto';
             userInput.style.height = `${userInput.scrollHeight}px`;
             userInput.focus();
             sendButton.disabled = true;
        } else {
             addMessage(messageText, 'user'); // Add user message bubble (from prompt button)
             userInput.focus(); // Still focus input
             sendButton.disabled = true; // Disable button
        }


        showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: messageText }),
            });
            hideTypingIndicator();

            if (!response.ok) {
                let errorText = `Error: ${response.status} ${response.statusText}`;
                try { const errorJson = await response.json(); errorText = errorJson.reply || errorText; } catch (e) { /* Ignore */ }
                addMessage(`Oops! Something went wrong. ${errorText}`, 'bot');
                return;
            }
            const data = await response.json();
            addMessage(data.reply, 'bot');
        } catch (error) {
            hideTypingIndicator();
            console.error('Fetch Error:', error);
            addMessage('Sorry, I encountered a problem connecting to the server.', 'bot');
        }
    }

    // --- Event Listeners ---
    sendButton.addEventListener('click', () => sendMessage()); // Use wrapper

    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            if (!sendButton.disabled) { sendMessage(); } // Use wrapper
        }
    });

    // Add listeners for example prompt buttons
    const promptButtons = document.querySelectorAll('.prompt-button');
    promptButtons.forEach(button => {
        button.addEventListener('click', () => {
            const promptText = button.textContent.replace(/"/g, ''); // Get text, remove quotes
            // Option 1: Fill input and submit
            // userInput.value = promptText;
            // userInput.dispatchEvent(new Event('input')); // Trigger resize/enable button
            // sendMessage();

            // Option 2: Directly send the prompt text
            sendMessage(promptText);
        });
    });


    // --- Initial setup ---
    userInput.focus();
    // Don't scroll initially if placeholder is visible
    // messagesDiv.scrollTop = messagesDiv.scrollHeight;
    sendButton.disabled = true;
    userInput.dispatchEvent(new Event('input'));

    // Check if placeholder should be visible (it should be if messagesDiv is initially empty apart from it and indicator)
    const existingMessages = messagesDiv.querySelectorAll('.message:not(.typing-indicator):not(#initial-placeholder)');
    if (existingMessages.length > 0) {
        hidePlaceholder(); // Hide if messages were loaded from history server-side perhaps
    } else {
         initialPlaceholder.style.display = 'flex'; // Ensure it's visible
         placeholderVisible = true;
    }


});