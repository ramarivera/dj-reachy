"""Custom Gradio theme for DJ Reachy to match the headless UI style."""

import gradio as gr

# DJ Reachy color palette (matching static/style.css)
DJ_REACHY_COLORS = {
    "bg": "#060b1a",
    "bg_2": "#071023",
    "panel": "rgba(11, 18, 36, 0.95)",
    "border": "rgba(255, 255, 255, 0.08)",
    "text": "#eaf2ff",
    "muted": "#9fb6d7",
    "ok": "#4ce0b3",
    "warn": "#ffb547",
    "error": "#ff5c70",
    "accent": "#45c4ff",
    "accent_2": "#5ef0c1",
}


def create_dj_reachy_theme() -> gr.themes.Base:
    """Create a custom Gradio theme matching DJ Reachy's design."""
    # Use Soft dark theme as base and customize colors
    return gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#e6f7ff",
            c100="#bae7ff",
            c200="#91d5ff",
            c300="#69c0ff",
            c400="#45c4ff",
            c500="#45c4ff",
            c600="#1890ff",
            c700="#096dd9",
            c800="#0050b3",
            c900="#003a8c",
            c950="#002766",
        ),
        secondary_hue=gr.themes.Color(
            c50="#e6fffb",
            c100="#b5f5ec",
            c200="#87e8de",
            c300="#5ef0c1",
            c400="#5ef0c1",
            c500="#4ce0b3",
            c600="#13c2c2",
            c700="#08979c",
            c800="#006d75",
            c900="#00474f",
            c950="#002329",
        ),
        neutral_hue=gr.themes.Color(
            c50="#eaf2ff",
            c100="#d5e5ff",
            c200="#9fb6d7",
            c300="#7a9bc5",
            c400="#5a7fb3",
            c500="#3d5a80",
            c600="#2d4a6e",
            c700="#1e3a5c",
            c800="#0f2a4a",
            c900="#071023",
            c950="#060b1a",
        ),
        font=("Space Grotesk", "Inter", "system-ui", "sans-serif"),
        font_mono=("JetBrains Mono", "Fira Code", "monospace"),
    )


# Custom CSS to enhance the theme
DJ_REACHY_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* Ambient background effect */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background: 
        radial-gradient(circle at 20% 20%, rgba(69, 196, 255, 0.12), transparent 35%),
        radial-gradient(circle at 80% 0%, rgba(94, 240, 193, 0.12), transparent 32%),
        radial-gradient(circle at 30% 60%, rgba(255, 255, 255, 0.03), transparent 35%);
    pointer-events: none;
    z-index: -1;
}

/* Main container */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    background: transparent !important;
}

/* Title styling */
.gradio-container h1 {
    background: linear-gradient(120deg, #45c4ff, #5ef0c1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Panel styling */
.panel, .gr-panel, .gr-box, .gr-form {
    background: rgba(11, 18, 36, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
}

/* Input fields */
input[type="text"], input[type="password"], textarea, select {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #eaf2ff !important;
    transition: all 0.15s ease !important;
}

input:focus, textarea:focus, select:focus {
    border-color: rgba(94, 240, 193, 0.7) !important;
    box-shadow: 0 0 0 3px rgba(94, 240, 193, 0.15) !important;
    outline: none !important;
}

/* Primary buttons */
.primary, button.primary, .gr-button-primary {
    background: linear-gradient(120deg, #45c4ff, #5ef0c1) !important;
    color: #031022 !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 14px 40px rgba(69, 196, 255, 0.25) !important;
    transition: all 0.15s ease !important;
}

.primary:hover, button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 18px 50px rgba(69, 196, 255, 0.35) !important;
    filter: brightness(1.05) !important;
}

/* Secondary/ghost buttons */
.secondary, button.secondary, .gr-button-secondary {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #eaf2ff !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.secondary:hover, button.secondary:hover, .gr-button-secondary:hover {
    border-color: rgba(94, 240, 193, 0.4) !important;
    background: rgba(255, 255, 255, 0.08) !important;
}

/* Chatbot styling */
.chatbot-container, .gr-chatbot {
    background: rgba(11, 18, 36, 0.7) !important;
    border-radius: 14px !important;
}

.message.user {
    background: rgba(69, 196, 255, 0.15) !important;
    border: 1px solid rgba(69, 196, 255, 0.3) !important;
}

.message.bot {
    background: rgba(94, 240, 193, 0.1) !important;
    border: 1px solid rgba(94, 240, 193, 0.2) !important;
}

/* Labels */
label, .gr-label {
    color: #9fb6d7 !important;
    font-size: 13px !important;
    letter-spacing: 0.2px !important;
}

/* Accordion styling */
.gr-accordion {
    background: rgba(11, 18, 36, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
}

/* Dropdown */
.gr-dropdown {
    background: rgba(255, 255, 255, 0.04) !important;
}

/* Status chips/badges */
.chip, .badge, .tag {
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.chip-ok, .badge-success {
    background: rgba(76, 224, 179, 0.15) !important;
    color: #4ce0b3 !important;
    border-color: rgba(76, 224, 179, 0.4) !important;
}

/* Audio component styling */
.audio-component, .gr-audio {
    background: rgba(11, 18, 36, 0.8) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* WebRTC component */
.webrtc-container {
    background: rgba(11, 18, 36, 0.9) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.15);
}

/* Footer styling */
footer {
    background: transparent !important;
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Tab styling */
.tabs, .gr-tabs {
    background: transparent !important;
}

.tab-nav button {
    background: transparent !important;
    color: #9fb6d7 !important;
    border-bottom: 2px solid transparent !important;
}

.tab-nav button.selected {
    color: #45c4ff !important;
    border-bottom-color: #45c4ff !important;
}

/* Markdown content */
.prose, .markdown {
    color: #eaf2ff !important;
}

.prose code, .markdown code {
    background: rgba(69, 196, 255, 0.1) !important;
    color: #45c4ff !important;
    padding: 2px 6px;
    border-radius: 4px;
}

/* Recording indicator */
.recording {
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Hero section for title */
.hero-title {
    text-align: center;
    margin-bottom: 24px;
}

.hero-title h1 {
    font-size: 2.5rem !important;
    margin-bottom: 8px !important;
}

.hero-subtitle {
    color: #9fb6d7;
    font-size: 1.1rem;
    margin: 0;
}

/* Connection status indicator */
.connection-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 13px;
    background: rgba(76, 224, 179, 0.1);
    border: 1px solid rgba(76, 224, 179, 0.3);
    color: #4ce0b3;
}

.connection-status.disconnected {
    background: rgba(255, 92, 112, 0.1);
    border-color: rgba(255, 92, 112, 0.3);
    color: #ff5c70;
}

/* API key section styling */
.api-key-section {
    background: rgba(11, 18, 36, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

.api-key-section.configured {
    border-color: rgba(76, 224, 179, 0.3);
}
"""

