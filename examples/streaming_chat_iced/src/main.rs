//! Streaming chat with Iced — chat bubbles, provider/model picker, RAG folder.
//!
//! ```sh
//! cargo run --manifest-path examples/streaming_chat_iced/Cargo.toml
//! ```
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`ChatAgentSpec::parse`] | Build a Generator by parsing a backend string (ollama/deepseek) |
//! | [`OllamaEmbedder::new`] | Embed queries/documents via local Ollama |
//! | [`open_store`] | Open an existing vector store on disk |
//! | [`search_similar`] | Run a similarity search over the vector store |
//! | [`collect_documents`] | Parse + embed + store all documents in a folder |
//! | [`DocumentParsers::new`] | Bundle all registered format parsers |
//! | [`build_parsers`] | Get the default set of document parsers |
//! | [`ChunkConfig`] | Configure chunk size and overlap |
//! | [`ScoredChunk`] | A chunk returned from similarity search with its score |
//! | [`Generator::generate_stream`] | Stream tokens from the LLM via a callback |
//! | [`Generator`] (trait) | The trait all chat backends implement |

use iced::widget::{button, column, container, pick_list, row, scrollable, text, text_input, Space};
use iced::{Alignment, Element, Fill, Length, Subscription, Task, Theme};
// ── ragrig imports ──
use ragrig::agents::ChatAgentSpec;       // parse a backend spec into a Generator
use ragrig::embed::OllamaEmbedder;       // embed queries/documents via local Ollama
use ragrig::parsers::{DocumentParsers, build_parsers};  // document parsing & chunking
use ragrig::store::{ScoredChunk, open_store};           // vector store & search results
use ragrig::types::ChunkConfig;          // chunk size/overlap configuration
use ragrig::vector::{collect_documents, search_similar}; // indexing & search
use std::path::PathBuf;

// ── Main entry ───────────────────────────────────────────────────────────

fn main() -> iced::Result {
    iced::application(RagChat::boot, RagChat::update, RagChat::view)
        .title("RAG Chat — ragrig + Iced")
        .subscription(RagChat::subscription)
        .theme(Theme::Dark)
        .centered()
        .run()
}

// ── Messages ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Message {
    InputChanged(String),
    SendMessage,
    ProviderSelected(ProviderChoice),
    ModelChanged(String),
    ApiKeyChanged(String),
    FolderSelected,
    FolderPicked(Option<PathBuf>),
    StreamToken(String),
    IndexDocuments,
    IndexDone(Result<usize, String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderChoice {
    Ollama,
    DeepSeek,
}

impl ProviderChoice {
    fn all() -> &'static [Self] {
        &[Self::Ollama, Self::DeepSeek]
    }
}

impl std::fmt::Display for ProviderChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ollama => write!(f, "Ollama"),
            Self::DeepSeek => write!(f, "DeepSeek"),
        }
    }
}

// ── Chat message bubble ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ChatBubble {
    role: BubbleRole,
    content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BubbleRole {
    User,
    Assistant,
    System,
}

// ── Application state ─────────────────────────────────────────────────────

struct RagChat {
    messages: Vec<ChatBubble>,
    input: String,
    is_streaming: bool,
    streaming_buffer: String,

    provider: ProviderChoice,
    model: String,
    api_key: String,

    folder_path: Option<PathBuf>,
    folder_input: String,
    chunk_count: usize,
}

impl RagChat {
    fn boot() -> (Self, Task<Message>) {
        (
            Self {
                messages: Vec::new(),
                input: String::new(),
                is_streaming: false,
                streaming_buffer: String::new(),
                provider: ProviderChoice::Ollama,
                model: String::from("gemma2:latest"),
                api_key: String::new(),
                folder_path: None,
                folder_input: String::new(),
                chunk_count: 0,
            },
            Task::none(),
        )
    }

    // ── Prompt building ───────────────────────────────────────────────

    fn build_prompt(user_query: &str, chunks: &[ScoredChunk]) -> String {
        if chunks.is_empty() {
            format!(
                "You are a helpful assistant. Answer the user's question.\n\nUser: {}\nAssistant:",
                user_query
            )
        } else {
            let mut ctx = String::from("Context from documents:\n\n");
            for (i, sc) in chunks.iter().enumerate() {
                ctx.push_str(&format!(
                    "--- Snippet {} (score: {:.2}) ---\n{}\n\n",
                    i + 1,
                    sc.score,
                    sc.chunk.text
                ));
            }
            format!(
                "You are a helpful document assistant. Answer the user's question \
                 explicitly using the provided Context snippets.\n\n{}\n\nUser: {}\nAssistant:",
                ctx, user_query
            )
        }
    }

    // ── Update ────────────────────────────────────────────────────────

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::InputChanged(s) => {
                self.input = s;
                Task::none()
            }

            Message::SendMessage => {
                if self.input.trim().is_empty() || self.is_streaming {
                    return Task::none();
                }
                let user_text = std::mem::take(&mut self.input);
                self.messages.push(ChatBubble {
                    role: BubbleRole::User,
                    content: user_text.clone(),
                });

                self.is_streaming = true;
                self.streaming_buffer = String::new();

                self.messages.push(ChatBubble {
                    role: BubbleRole::Assistant,
                    content: String::from("..."),
                });

                let provider = self.provider;
                let model = self.model.clone();
                let api_key = self.api_key.clone();
                let folder = self.folder_path.clone();

                let (tx, rx) = iced::futures::channel::mpsc::unbounded::<String>();

                tokio::spawn(async move {
                    let backend = match provider {
                        ProviderChoice::Ollama => "ollama",
                        ProviderChoice::DeepSeek => "deepseek",
                    };
                    let key_opt = if api_key.is_empty() {
                        None
                    } else {
                        Some(api_key.as_str())
                    };

                    // ── ragrig: parse backend string into a ChatAgentSpec ──
                    let spec = match ChatAgentSpec::parse(backend, Some(&model), key_opt, None) {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = tx.unbounded_send(format!("Error: {}", e));
                            return;
                        }
                    };

                    // ── ragrig: build the Generator from the spec ──
                    let generator = match spec.build() {
                        Ok(g) => g,
                        Err(e) => {
                            let _ = tx.unbounded_send(format!("Error: {}", e));
                            return;
                        }
                    };

                    let chunks: Vec<ScoredChunk> = if let Some(ref f) = folder {
                        // ── ragrig: open the vector store for the selected folder ──
                        match open_store(f).await {
                            Ok(store) => {
                                // ── ragrig: create Ollama embedder ──
                                let emb = OllamaEmbedder::new(
                                    "nomic-embed-text".to_string(),
                                );
                                // ── ragrig: search for similar chunks ──
                                match search_similar(&emb, 10, 0.4, &*store, &user_text)
                                    .await
                                {
                                    Ok(c) => c,
                                    Err(_) => Vec::new(),
                                }
                            }
                            Err(_) => Vec::new(),
                        }
                    } else {
                        Vec::new()
                    };

                    // ── ragrig: build prompt with document context ──
                    let prompt = Self::build_prompt(&user_text, &chunks);

                    let tx2 = tx.clone();
                    // ── ragrig: stream tokens from the LLM through a callback ──
                    let result = generator
                        .generate_stream(&prompt, &move |token| {
                            let _ = tx2.unbounded_send(token);
                        })
                        .await;

                    if let Err(e) = result {
                        let _ = tx.unbounded_send(format!("\n\nError: {}", e));
                    }
                    // Signal completion
                    let _ = tx.unbounded_send(String::from("__DONE__"));
                });

                Task::run(rx, Message::StreamToken)
            }

            Message::StreamToken(token) => {
                if token == "__DONE__" {
                    self.is_streaming = false;
                    return Task::none();
                }
                self.streaming_buffer.push_str(&token);
                if let Some(last) = self.messages.last_mut() {
                    if last.role == BubbleRole::Assistant {
                        last.content = self.streaming_buffer.clone();
                    }
                }
                Task::none()
            }

            Message::ProviderSelected(choice) => {
                self.provider = choice;
                self.model = match choice {
                    ProviderChoice::Ollama => "gemma2:latest".to_string(),
                    ProviderChoice::DeepSeek => "deepseek-chat".to_string(),
                };
                Task::none()
            }

            Message::ModelChanged(s) => {
                self.model = s;
                Task::none()
            }

            Message::ApiKeyChanged(s) => {
                self.api_key = s;
                Task::none()
            }

            Message::FolderSelected => Task::perform(
                async {
                    let path = rfd::AsyncFileDialog::new()
                        .pick_folder()
                        .await
                        .map(|h| h.path().to_path_buf());
                    Message::FolderPicked(path)
                },
                |m| m,
            ),

            Message::FolderPicked(opt_path) => {
                if let Some(path) = opt_path {
                    self.folder_path = Some(path.clone());
                    self.folder_input = path.display().to_string();
                    return self.start_indexing();
                }
                Task::none()
            }

            Message::IndexDocuments => self.start_indexing(),

            Message::IndexDone(result) => {
                match result {
                    Ok(count) => {
                        self.chunk_count = count;
                        self.messages.push(ChatBubble {
                            role: BubbleRole::System,
                            content: format!("Indexed {} chunks from documents.", count),
                        });
                    }
                    Err(e) => {
                        self.messages.push(ChatBubble {
                            role: BubbleRole::System,
                            content: format!("Indexing error: {}", e),
                        });
                    }
                }
                Task::none()
            }
        }
    }

    fn start_indexing(&self) -> Task<Message> {
        let folder = match &self.folder_path {
            Some(p) => p.clone(),
            None => {
                return Task::done(Message::IndexDone(Err(
                    "No folder selected".to_string(),
                )));
            }
        };

        // ── ragrig: configure chunk size and overlap ──
        let config = ChunkConfig {
            size: 1024,
            overlap: 128,
        };

        Task::perform(
            async move {
                // ── ragrig: create embedder for document indexing ──
                let embedder = OllamaEmbedder::new("nomic-embed-text".to_string());
                // ── ragrig: open the vector store ──
                let store = open_store(&folder)
                    .await
                    .map_err(|e| format!("Failed to open vector store: {}", e))?;

                // ── ragrig: bundle document parsers ──
                let parsers = DocumentParsers::new(build_parsers());
                // ── ragrig: parse, embed, and store all documents ──
                let _ = collect_documents(&embedder, &parsers, &folder, &config, &*store)
                    .await
                    .map_err(|e| format!("Indexing failed: {}", e))?;

                // ── ragrig: return chunk count from the store ──
                Ok(store.len())
            },
            Message::IndexDone,
        )
    }

    // ── Subscription ──────────────────────────────────────────────────

    fn subscription(&self) -> Subscription<Message> {
        Subscription::none()
    }

    // ── View ──────────────────────────────────────────────────────────

    fn view(&self) -> Element<'_, Message> {
        let top_bar = self.view_top_bar();
        let chat_area = self.view_chat_area();
        let input_bar = self.view_input_bar();

        column![top_bar, chat_area, input_bar]
            .width(Fill)
            .height(Fill)
            .padding(12)
            .spacing(8)
            .into()
    }

    fn view_top_bar(&self) -> Element<'_, Message> {
        let provider_pick = pick_list(
            ProviderChoice::all(),
            Some(self.provider),
            Message::ProviderSelected,
        )
        .placeholder("Select provider...");

        let model_input = text_input("Model name", &self.model)
            .on_input(Message::ModelChanged)
            .width(180);

        let mut top_row = row![
            text("Provider:").size(14),
            provider_pick,
            Space::new().width(16),
            text("Model:").size(14),
            model_input,
        ]
        .align_y(Alignment::Center)
        .spacing(6);

        if self.provider == ProviderChoice::DeepSeek {
            top_row = top_row
                .push(Space::new().width(12))
                .push(
                    text_input("DeepSeek API key (or set env)", &self.api_key)
                        .on_input(Message::ApiKeyChanged)
                        .width(220)
                        .secure(true),
                );
        }

        let folder_input = text_input("Document folder...", &self.folder_input)
            .width(200);

        let folder_btn = button("📁 Browse").on_press(Message::FolderSelected);

        let index_btn = button(text("🔄 Re-index").size(12))
            .on_press(Message::IndexDocuments);

        let mut docs_row = row![
            Space::new().width(Fill),
            text("Docs:").size(14),
            folder_input,
            folder_btn,
            index_btn,
        ]
        .align_y(Alignment::Center)
        .spacing(6);

        if self.chunk_count > 0 {
            docs_row = docs_row.push(
                text(format!("({} chunks)", self.chunk_count))
                    .size(12)
                    .style(text::secondary),
            );
        }

        let full_top = row![top_row, docs_row].align_y(Alignment::Center);

        container(full_top)
            .width(Fill)
            .padding(8)
            .style(container::rounded_box)
            .into()
    }

    fn view_chat_area(&self) -> Element<'_, Message> {
        let bubbles: Vec<Element<'_, Message>> = self
            .messages
            .iter()
            .map(|msg| self.view_bubble(msg))
            .collect();

        let col = column(bubbles)
            .width(Fill)
            .spacing(8)
            .padding(8);

        scrollable(col)
            .width(Fill)
            .height(Fill)
            .id(iced::widget::Id::new("chat"))
            .into()
    }

    fn view_bubble<'a>(&'a self, msg: &'a ChatBubble) -> Element<'a, Message> {
        let (alignment, color_bg, color_text, label) = match msg.role {
            BubbleRole::User => (
                iced::alignment::Horizontal::Right,
                [0.18, 0.40, 0.80],
                [1.0, 1.0, 1.0],
                "You",
            ),
            BubbleRole::Assistant => (
                iced::alignment::Horizontal::Left,
                [0.22, 0.22, 0.24],
                [0.9, 0.9, 0.9],
                "Assistant",
            ),
            BubbleRole::System => (
                iced::alignment::Horizontal::Center,
                [0.15, 0.15, 0.18],
                [0.6, 0.6, 0.6],
                "System",
            ),
        };

        let header = text(label).size(11).style(move |_theme| {
            text::Style {
                color: Some(iced::Color::from_linear_rgba(
                    color_text[0] * 0.7,
                    color_text[1] * 0.7,
                    color_text[2] * 0.7,
                    0.8,
                )),
            }
        });

        let body = text(&msg.content).size(14).style(move |_theme| {
            text::Style {
                color: Some(iced::Color::from_linear_rgba(
                    color_text[0],
                    color_text[1],
                    color_text[2],
                    1.0,
                )),
            }
        });

        let bubble = column![header, body]
            .width(Length::Shrink)
            .max_width(480)
            .padding(10)
            .spacing(4);

        let bubble_container = container(bubble).style(move |_theme| {
            container::Style {
                background: Some(
                    iced::Color::from_linear_rgba(
                        color_bg[0],
                        color_bg[1],
                        color_bg[2],
                        0.85,
                    )
                    .into(),
                ),
                border: iced::Border {
                    radius: 12.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        });

        container(bubble_container)
            .align_x(alignment)
            .width(Fill)
            .into()
    }

    fn view_input_bar(&self) -> Element<'_, Message> {
        let input = text_input("Type a message...", &self.input)
            .on_input(Message::InputChanged)
            .on_submit(Message::SendMessage)
            .width(Fill)
            .padding(8);

        let send_label = if self.is_streaming { "⏳" } else { "Send" };
        let send_btn = button(text(send_label).size(14))
            .on_press_maybe(
                if self.is_streaming || self.input.trim().is_empty() {
                    None
                } else {
                    Some(Message::SendMessage)
                },
            )
            .padding([8, 16]);

        container(row![input, send_btn].align_y(Alignment::Center).spacing(8))
            .width(Fill)
            .padding(8)
            .style(container::rounded_box)
            .into()
    }
}
