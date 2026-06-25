//! Streaming chat with egui — two-color bubbles, auto-scroll, dark mode.
//!
//! ```sh
//! cargo run
//! ```
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`ChatAgentSpec::ollama`] | Build a Generator from an Ollama model spec |
//! | [`Generator::generate_stream`] | Stream tokens from the LLM via a callback |
//! | [`Generator`] (trait) | The trait all chat backends implement |

use eframe::egui;
// ── ragrig import: the Generator trait and ChatAgentSpec builder ──
use ragrig::agents::Generator;
use std::sync::Arc;

struct ChatMessage {
    role: String,
    content: String,
    cache: egui_commonmark::CommonMarkCache,
}

struct ChatApp {
    messages: Vec<ChatMessage>,
    input: String,
    runtime: tokio::runtime::Runtime,
    agent: Arc<Box<dyn Generator>>,
    stream_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    current: String,
    streaming: bool,
    send_pending: bool,
    streaming_cache: egui_commonmark::CommonMarkCache,
}

impl ChatApp {
    fn new() -> Self {
        // ── ragrig: build a Generator from an Ollama model spec ──
        let agent = ragrig::agents::ChatAgentSpec::ollama("gemma2:latest", Default::default())
            .build().unwrap();
        Self {
            messages: Vec::new(),
            input: String::new(),
            runtime: tokio::runtime::Runtime::new().unwrap(),
            agent: Arc::new(agent),
            stream_rx: None,
            current: String::new(),
            streaming: false,
            send_pending: false,
            streaming_cache: egui_commonmark::CommonMarkCache::default(),
        }
    }

    fn try_send(&mut self) {
        if !self.send_pending || self.streaming || self.input.trim().is_empty() { return; }
        self.send_pending = false;
        let text = std::mem::take(&mut self.input);
        self.messages.push(ChatMessage {
            role: "User".into(),
            content: text.clone(),
            cache: egui_commonmark::CommonMarkCache::default(),
        });
        self.streaming = true;
        self.current.clear();

        let prompt = self.messages.iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>().join("\n") + "\nAssistant: ";

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.stream_rx = Some(rx);
        let agent = self.agent.clone();
        self.runtime.spawn(async move {
            // ── ragrig: stream tokens from the LLM through a callback channel ──
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
                        self.messages.push(ChatMessage {
                            role: "Assistant".into(),
                            content: response,
                            cache: std::mem::take(&mut self.streaming_cache),
                        });
                    }
                    break;
                }
            }
        }
    }
}

fn render_bubble(ui: &mut egui::Ui, text: &str, cache: &mut egui_commonmark::CommonMarkCache, is_user: bool, max_w: f32) {
    let is_dark = ui.visuals().dark_mode;
    let (fill, text_color) = if is_user {
        if is_dark { (egui::Color32::from_rgb(37, 99, 235), egui::Color32::WHITE) }
        else { (egui::Color32::from_rgb(219, 234, 254), egui::Color32::BLACK) }
    } else {
        if is_dark { (egui::Color32::from_rgb(64, 65, 79), egui::Color32::from_gray(236)) }
        else { (egui::Color32::from_rgb(243, 244, 246), egui::Color32::BLACK) }
    };
    let frame = egui::Frame::new()
        .fill(fill)
        .corner_radius(12)
        .inner_margin(egui::Margin::symmetric(12, 8))
        .outer_margin(egui::Margin::symmetric(0, 4));

    let render = |ui: &mut egui::Ui| {
        ui.set_max_width(max_w);
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
        ui.visuals_mut().override_text_color = Some(text_color);
        ui.visuals_mut().hyperlink_color = if is_user && is_dark {
            egui::Color32::from_rgb(147, 197, 253)
        } else { text_color };
        egui_commonmark::CommonMarkViewer::new().show(ui, cache, text);
        ui.visuals_mut().override_text_color = None;
    };

    if is_user {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            frame.show(ui, render);
        });
    } else {
        frame.show(ui, render);
    }
}

impl eframe::App for ChatApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_stream();
        self.try_send();
        if self.streaming { ctx.request_repaint(); }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let bubble_w = (ui.available_width() * 0.78).min(680.0);

        // ── Messages ──────────────────────────────────────────
        let bottom_h = 48.0;
        egui::ScrollArea::vertical()
            .max_height(ui.available_height() - bottom_h - 8.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.messages.is_empty() && !self.streaming {
                    ui.add_space(60.0);
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("Start a conversation").size(20.0)
                            .color(egui::Color32::from_gray(160)));
                    });
                }
                for msg in &mut self.messages {
                    render_bubble(ui, &msg.content, &mut msg.cache, msg.role == "User", bubble_w);
                }
                if self.streaming {
                    let fill = egui::Color32::from_rgb(64, 65, 79);
                    let frame = egui::Frame::new().fill(fill).corner_radius(12)
                        .inner_margin(egui::Margin::symmetric(12, 8))
                        .outer_margin(egui::Margin::symmetric(0, 4));
                    frame.show(ui, |ui| {
                        ui.set_max_width(bubble_w);
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                        ui.visuals_mut().override_text_color = Some(egui::Color32::from_gray(236));
                        egui_commonmark::CommonMarkViewer::new().show(ui, &mut self.streaming_cache, &self.current);
                        ui.add(egui::Spinner::new().size(14.0));
                        ui.visuals_mut().override_text_color = None;
                    });
                }
            });

        ui.separator();

        // ── Input ─────────────────────────────────────────────
        ui.horizontal(|ui| {
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.input)
                    .hint_text("Type a message...")
                    .desired_width(f32::INFINITY),
            );
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let send = ui.add_enabled(
                !self.input.trim().is_empty() && !self.streaming,
                egui::Button::new("Send"),
            ).clicked();
            if enter || send {
                self.send_pending = true;
                resp.request_focus();
            }
        });
    }
}

fn main() -> eframe::Result {
    eframe::run_native(
        "ragrig Chat",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(ChatApp::new()))
        }),
    )
}
