//! Streaming chat with ratatui — two-color bubbles, scroll, word wrap.
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
use ragrig::agents::Generator;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    widgets::{Block, Clear, Paragraph, Wrap},
    Frame,
};
use std::sync::Arc;

struct ChatApp {
    messages: Vec<(String, String)>,
    input: String,
    runtime: tokio::runtime::Runtime,
    agent: Arc<Box<dyn Generator>>,
    stream_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    current: String,
    streaming: bool,
    quit: bool,
    scroll_offset: usize,
    pinned_to_bottom: bool,
}

impl ChatApp {
    fn new() -> Self {
        let spec = ragrig::agents::ChatAgentSpec::ollama("gemma2:latest", Default::default());
        Self {
            messages: Vec::new(),
            input: String::new(),
            runtime: tokio::runtime::Runtime::new().unwrap(),
            agent: Arc::new(spec.build().unwrap()),
            stream_rx: None,
            current: String::new(),
            streaming: false,
            quit: false,
            scroll_offset: 0,
            pinned_to_bottom: true,
        }
    }

    fn try_send(&mut self) {
        if self.streaming || self.input.trim().is_empty() { return; }
        let text = std::mem::take(&mut self.input);
        self.messages.push(("User".into(), text.clone()));
        self.streaming = true;
        self.current.clear();
        self.scroll_offset = usize::MAX;
        self.pinned_to_bottom = true;

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
                Ok(t) => {
                    self.current.push_str(&t);
                    if self.pinned_to_bottom { self.scroll_offset = usize::MAX; }
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    self.streaming = false;
                    self.stream_rx = None;
                    let response = std::mem::take(&mut self.current);
                    if !response.is_empty() {
                        self.messages.push(("Assistant".into(), response));
                        if self.pinned_to_bottom { self.scroll_offset = usize::MAX; }
                    }
                    break;
                }
            }
        }
    }
}

fn word_wrap(text: &str, max_width: u16) -> Vec<String> {
    let max_width = max_width as usize;
    if max_width < 2 { return text.lines().map(String::from).collect(); }
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() { lines.push(String::new()); continue; }
        let mut current = String::new();
        for word in paragraph.split(' ') {
            if current.is_empty() { current.push_str(word); }
            else if current.len() + 1 + word.len() <= max_width { current.push(' '); current.push_str(word); }
            else {
                if !current.is_empty() { lines.push(current.clone()); }
                current.clear(); current.push_str(word);
            }
        }
        if !current.is_empty() { lines.push(current); }
    }
    lines
}

fn bubble_height(text: &str, max_width: u16) -> u16 {
    let inner = max_width.saturating_sub(2);
    let lines = word_wrap(text, inner);
    lines.len().max(1) as u16 + 2
}

fn render_bubble(f: &mut Frame, text: &str, area: Rect, is_user: bool) {
    let border_color = if is_user { Color::Rgb(37, 99, 235) } else { Color::Rgb(64, 65, 79) };
    let bg_color = if is_user { Color::Rgb(30, 58, 138) } else { Color::Rgb(55, 55, 65) };
    let bubble = Paragraph::new(text)
        .block(Block::bordered().border_style(Style::new().fg(border_color)))
        .style(Style::new().bg(bg_color).fg(Color::Rgb(236, 236, 236)))
        .wrap(Wrap { trim: false });
    f.render_widget(Clear, area);
    f.render_widget(bubble, area);
}

fn render_streaming_bubble(f: &mut Frame, text: &str, area: Rect) {
    let content = if text.is_empty() { "▊".to_string() } else { format!("{}▊", text) };
    let bg_color = Color::Rgb(55, 55, 65);
    let border_color = Color::Rgb(64, 65, 79);
    let bubble = Paragraph::new(content)
        .block(Block::bordered().border_style(Style::new().fg(border_color)))
        .style(Style::new().bg(bg_color).fg(Color::Rgb(236, 236, 236)))
        .wrap(Wrap { trim: false });
    f.render_widget(Clear, area);
    f.render_widget(bubble, area);
}

fn render_messages(f: &mut Frame, app: &mut ChatApp, area: Rect) {
    let bubble_max_width = ((area.width as f32) * 0.78) as u16;
    let mut heights: Vec<u16> = Vec::new();
    for (_, content) in &app.messages { heights.push(bubble_height(content, bubble_max_width) + 1); }
    if app.streaming { heights.push(bubble_height(&app.current, bubble_max_width) + 1); }
    let total_height: usize = heights.iter().map(|h| *h as usize).sum::<usize>();
    let viewport_h = area.height as usize;

    if total_height == 0 && !app.streaming {
        let empty = Paragraph::new("Start a conversation")
            .style(Style::new().fg(Color::DarkGray));
        let vert_center = Rect { y: area.y + area.height.saturating_sub(1) / 2, height: 1, ..area };
        f.render_widget(empty, vert_center);
        return;
    }
    if total_height > viewport_h {
        let max_scroll = total_height - viewport_h;
        if app.scroll_offset > max_scroll { app.scroll_offset = max_scroll; app.pinned_to_bottom = true; }
    } else { app.scroll_offset = 0; app.pinned_to_bottom = true; }

    let scroll = app.scroll_offset;
    let mut virtual_y: usize = 0;
    let mut idx: usize = 0;
    let n_items = heights.len();

    while idx < n_items {
        let h = heights[idx] as usize;
        let item_bottom = virtual_y + h;
        if item_bottom <= scroll { virtual_y = item_bottom; idx += 1; continue; }
        if virtual_y >= scroll + viewport_h { break; }
        let screen_y = area.y + virtual_y.saturating_sub(scroll) as u16;
        let visible_h = h.min(viewport_h.saturating_sub(virtual_y.saturating_sub(scroll))) as u16;
        if visible_h > 0 && screen_y < area.y + area.height {
            let is_streaming_item = app.streaming && idx == n_items - 1;
            let content = if is_streaming_item {
                if app.current.is_empty() { "▊" } else { &app.current }
            } else {
                &app.messages[idx].1
            };
            let bw = (content.len().max(4) as u16 + 4).min(bubble_max_width);
            let is_user = if is_streaming_item { false } else { app.messages[idx].0 == "User" };
            let x = if is_user { area.x + area.width.saturating_sub(bw) } else { area.x };
            let bubble_rect = Rect { x, y: screen_y, width: bw, height: visible_h };
            if bubble_rect.y < area.y + area.height
                && bubble_rect.y + bubble_rect.height <= area.y + area.height
            {
                if is_streaming_item {
                    render_streaming_bubble(f, &app.current, bubble_rect);
                } else {
                    render_bubble(f, &app.messages[idx].1, bubble_rect, is_user);
                }
            }
        }
        virtual_y = item_bottom;
        idx += 1;
    }
}

fn render_input(f: &mut Frame, app: &ChatApp, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Fill(1), Constraint::Length(6)])
        .split(area);
    let cursor = if !app.streaming { "▌" } else { "" };
    let display_text = format!("{}{}", app.input, cursor);
    let input = Paragraph::new(display_text)
        .block(Block::bordered().border_style(Style::new().fg(Color::Rgb(37, 99, 235))))
        .style(Style::new().fg(Color::White).bg(Color::Rgb(30, 30, 40)));
    f.render_widget(input, chunks[0]);
    let send = Paragraph::new(" Send ")
        .block(Block::bordered().border_style(Style::new().fg(Color::DarkGray)))
        .style(if !app.input.trim().is_empty() && !app.streaming {
            Style::new().fg(Color::White).bg(Color::Rgb(37, 99, 235))
        } else { Style::new().fg(Color::DarkGray) })
        .centered();
    f.render_widget(send, chunks[1]);
}

fn ui(f: &mut Frame, app: &mut ChatApp) {
    let area = f.area();
    f.render_widget(Clear, area);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Fill(1), Constraint::Length(3)])
        .split(area);

    let header = Paragraph::new(format!("ragrig Chat — {}", app.agent.model_name()))
        .style(Style::new().bold().fg(Color::White));
    f.render_widget(header, chunks[0]);
    render_messages(f, app, chunks[1]);
    render_input(f, app, chunks[2]);
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
        terminal.draw(|f| ui(f, &mut app))?;

        if app.quit { break; }
        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press || key.kind == KeyEventKind::Repeat {
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => app.quit = true,
                        KeyCode::Enter => { app.try_send(); }
                        KeyCode::Up => { app.scroll_offset = app.scroll_offset.saturating_sub(3); app.pinned_to_bottom = false; }
                        KeyCode::Down => { app.scroll_offset = app.scroll_offset.saturating_add(3); }
                        KeyCode::PageUp => { app.scroll_offset = app.scroll_offset.saturating_sub(20); app.pinned_to_bottom = false; }
                        KeyCode::PageDown => { app.scroll_offset = app.scroll_offset.saturating_add(20); }
                        KeyCode::Home => { app.scroll_offset = 0; app.pinned_to_bottom = false; }
                        KeyCode::End => { app.scroll_offset = usize::MAX; app.pinned_to_bottom = true; }
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
