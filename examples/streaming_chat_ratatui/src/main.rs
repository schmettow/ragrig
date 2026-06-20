//! Minimal streaming chat with ratatui.
//!
//! ```sh
//! cargo run
//! ```

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Paragraph},
    Frame,
};
use ragrig::agents::Generator;

struct ChatApp {
    messages: Vec<(String, String)>,
    input: String,
    runtime: tokio::runtime::Runtime,
    agent: std::sync::Arc<Box<dyn Generator>>,
    stream_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    current: String,
    streaming: bool,
    quit: bool,
}

impl ChatApp {
    fn new() -> Self {
        let spec = ragrig::agents::ChatAgentSpec::Ollama { model: "gemma2:latest".into() };
        Self {
            messages: Vec::new(),
            input: String::new(),
            runtime: tokio::runtime::Runtime::new().unwrap(),
            stream_rx: None,
            current: String::new(),
            streaming: false,
            quit: false,
            agent: std::sync::Arc::new(spec.build().unwrap()),
        }
    }

    fn try_send(&mut self) {
        if self.streaming || self.input.trim().is_empty() { return; }
        let text = std::mem::take(&mut self.input);
        self.messages.push(("User".into(), text.clone()));
        self.streaming = true;
        self.current.clear();

        let prompt = self.messages.iter()
            .map(|(r, c)| format!("{r}: {c}"))
            .collect::<Vec<_>>().join("\n") + "\nAssistant: ";

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.stream_rx = Some(rx);
        let agent = self.agent.clone();
        self.runtime.spawn(async move {
            let _ = agent.generate_stream(&prompt, &|t: String| { let _ = tx.send(t); }).await;
        });
    }

    fn poll_stream(&mut self) {
        let Some(ref mut rx) = self.stream_rx else { return };
        loop {
            match rx.try_recv() {
                Ok(t) => self.current.push_str(&t),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    self.streaming = false;
                    self.stream_rx = None;
                    let response = std::mem::take(&mut self.current);
                    if !response.is_empty() {
                        self.messages.push(("Assistant".into(), response));
                    }
                    break;
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut ChatApp) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Fill(1), Constraint::Length(3)])
        .split(area);

    // Messages
    let mut lines: Vec<Line> = Vec::new();
    for (role, content) in &app.messages {
        lines.push(Line::from(format!("{role}: {content}")));
    }
    if app.streaming {
        lines.push(Line::from(format!("Assistant: {}▊", app.current)));
    }
    let msgs = Paragraph::new(lines).block(Block::bordered().title("ragrig Chat"));
    f.render_widget(msgs, chunks[0]);

    // Input
    let input_text = format!("> {}{}", app.input, if app.streaming { "" } else { "▌" });
    let input = Paragraph::new(input_text)
        .block(Block::bordered().title("Message (Enter to send, ^C to quit)"))
        .style(Style::new().fg(Color::White));
    f.render_widget(input, chunks[1]);
}

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = ChatApp::new();

    loop {
        app.poll_stream();
        app.try_send();
        terminal.draw(|f| ui(f, &mut app))?;

        if app.quit { break; }
        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => app.quit = true,
                        KeyCode::Enter => { app.try_send(); }
                        KeyCode::Char(c) => app.input.push(c),
                        KeyCode::Backspace => { app.input.pop(); }
                        _ => {}
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
