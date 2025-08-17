// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles, newChatButton, themeToggle;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    newChatButton = document.getElementById('newChatButton');
    themeToggle = document.getElementById('themeToggle');
    
    initializeTheme();
    setupEventListeners();
    updateThemeToggleLabel();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // New chat functionality
    newChatButton.addEventListener('click', clearCurrentChat);
    
    // Theme toggle functionality
    themeToggle.addEventListener('click', toggleTheme);
    
    // Keyboard navigation for theme toggle
    themeToggle.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleTheme();
        }
    });
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        // Process sources to create clickable links
        const processedSources = sources.map(source => {
            // Check if source has embedded link
            const linkMatch = source.match(/^(.+?) \[LINK:(.+?)\]$/);
            if (linkMatch) {
                const [, sourceName, url] = linkMatch;
                return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="source-link">${sourceName}</a>`;
            }
            return escapeHtml(source);
        });
        
        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources</summary>
                <div class="sources-content">${processedSources.join(', ')}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

async function clearCurrentChat() {
    // Clear session on backend if we have one
    if (currentSessionId) {
        try {
            await fetch(`${API_URL}/sessions/${currentSessionId}/clear`, {
                method: 'DELETE'
            });
        } catch (error) {
            console.warn('Failed to clear session on backend:', error);
        }
    }
    
    // Reset frontend state
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
    
    // Focus the input field
    if (chatInput) {
        chatInput.focus();
    }
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}

// Theme Management Functions
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // Use saved theme if available, otherwise use system preference (defaulting to dark)
    const initialTheme = savedTheme || (prefersDarkMode ? 'dark' : 'light');
    
    // Apply theme using both class and data-theme attribute for flexibility
    setTheme(initialTheme);
    
    // Save the initial theme if not already saved
    if (!savedTheme) {
        localStorage.setItem('theme', initialTheme);
    }
}

function setTheme(theme) {
    const documentElement = document.documentElement;
    const body = document.body;
    
    // Remove existing theme classes and data attributes
    documentElement.classList.remove('light-theme', 'dark-theme');
    documentElement.removeAttribute('data-theme');
    if (body) {
        body.removeAttribute('data-theme');
    }
    
    // Apply new theme
    if (theme === 'light') {
        documentElement.classList.add('light-theme');
        documentElement.setAttribute('data-theme', 'light');
        if (body) {
            body.setAttribute('data-theme', 'light');
        }
    } else {
        documentElement.classList.add('dark-theme');
        documentElement.setAttribute('data-theme', 'dark');
        if (body) {
            body.setAttribute('data-theme', 'dark');
        }
    }
}

function toggleTheme() {
    const isLightMode = document.documentElement.classList.contains('light-theme');
    
    const newTheme = isLightMode ? 'dark' : 'light';
    
    // Apply the new theme with smooth transition
    setTheme(newTheme);
    
    // Save the preference
    localStorage.setItem('theme', newTheme);
    
    // Update aria-label for accessibility
    updateThemeToggleLabel();
    
    // Trigger a custom event for theme change
    document.dispatchEvent(new CustomEvent('themeChanged', {
        detail: { theme: newTheme }
    }));
}

function updateThemeToggleLabel() {
    const isLightMode = document.documentElement.classList.contains('light-theme');
    const newLabel = isLightMode 
        ? 'Switch to dark theme' 
        : 'Switch to light theme';
    
    if (themeToggle) {
        themeToggle.setAttribute('aria-label', newLabel);
        themeToggle.setAttribute('title', newLabel);
    }
}

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    // Only auto-switch if user hasn't manually set a preference
    const savedTheme = localStorage.getItem('theme');
    if (!savedTheme) {
        const systemTheme = e.matches ? 'dark' : 'light';
        setTheme(systemTheme);
        updateThemeToggleLabel();
    }
});