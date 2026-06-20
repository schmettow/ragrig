//! Minimal streaming chat with egui.
//!
//! ```sh
//! cargo run
//! ```

use eframe::egui;
use ragrig::agents::Generator;

struct ChatApp {
    messages: Vec<(String, String)>, // (role, content)
    input: String,
    runtime: tokio::runtime::Runtime,
    agent: std::sync::Arc<Box<dyn Generator>>,
    stream_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    current: String,
    streaming: bool,
}

impl ChatApp {
    fn new() -> Self {
        let spec = ragrig::agents::ChatAgentSpec::Ollama {
            model: "gemma2:latest".into(),
        };
        Self {
            messages: Vec::new(),
            input: String::new(),
            runtime: tokio::runtime::Runtime::new().unwrap(),
            agent: std::sync::Arc::new(spec.build().unwrap()),
            stream_rx: None,
            current: String::new(),
            streaming: false,
        }
    }

    fn try_send(&mut self) {
        if self.streaming || self.input.trim().is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.input);
        self.messages.push(("User".into(), text.clone()));
        self.streaming = true;
        self.current.clear();

        let prompt = self.messages.iter()
            .map(|(r, c)| format!("{r}: {c}"))
            .collect::<Vec<_>>()
            .join("\n") + "\nAssistant: ";

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.stream_rx = Some(rx);
        let agent = self.agent.clone();

        self.runtime.spawn(async move {
            let _ = agent.generate_stream(&prompt, &|t: String| {
                let _ = tx.send(t);
            }).await;
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

impl eframe::App for ChatApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_stream();
        self.try_send();
        if self.streaming {
            ctx.request_repaint();
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("ragrig Chat");

        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for (role, content) in &self.messages {
                    ui.label(egui::RichText::new(format!("{role}: {content}")).strong());
                }
                if self.streaming {
                    ui.label(format!("Assistant: {}▊", self.current));
                }
            });

        ui.separator();

        ui.horizontal(|ui| {
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.input)
                    .hint_text("Type a message...")
                    .desired_width(f32::INFINITY),
            );
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let send = ui.button("Send").clicked();
            if enter || send {
                self.try_send();
                resp.request_focus();
            }
        });
    }
}

fn main() -> eframe::Result {
    eframe::run_native(
        "ragrig Chat",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(ChatApp::new()))),
    )
}
