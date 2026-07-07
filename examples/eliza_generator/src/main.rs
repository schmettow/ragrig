//! Eliza chatbot (Weizenbaum 1966) as a custom [`ragrig::Generator`] backend.
//!
//! Demonstrates the [`SimpleGenerator`] trait — implement three methods,
//! wrap in [`MutexGenerator`], and you have a full [`Generator`] for free.
//! No `clone_box`, no `generate_stream`.
//!
//! ```sh
//! cargo run --manifest-path examples/eliza_generator/Cargo.toml
//! ```
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`SimpleGenerator`] | Trait for non-streaming backends (3 methods instead of 6) |
//! | [`MutexGenerator`] | Blanket adaptor: `SimpleGenerator` → `Generator` |
//! | [`Generator::generate`] | Call the backend through the standard trait |

use anyhow::Result;
use ragrig::agents::{Generator, MutexGenerator, SimpleGenerator};
use std::fmt;
use std::io::{BufRead, Write, stdin, stdout};

// ── Wrapper ────────────────────────────────────────────────────────────────

/// Wraps the `eliza` crate behind the [`SimpleGenerator`] trait.
///
/// `eliza::Eliza` doesn't implement `Debug` or `Clone`, so we provide
/// manual impls.  `MutexGenerator` handles the `&mut self` → `&self`
/// bridging and provides `clone_box` automatically.
struct ElizaBackend {
    inner: eliza::Eliza,
}

impl ElizaBackend {
    fn new() -> Self {
        let script = include_str!("../doctor.json");
        ElizaBackend {
            inner: eliza::Eliza::from_str(script)
                .expect("embedded DOCTOR script is valid JSON"),
        }
    }
}

// `eliza::Eliza` isn't Clone — re-create from the embedded script.
// MutexGenerator wraps us in Arc, so clone_box rarely calls this.
impl Clone for ElizaBackend {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl fmt::Debug for ElizaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElizaBackend").finish()
    }
}

impl SimpleGenerator for ElizaBackend {
    fn respond(&mut self, prompt: &str) -> String {
        self.inner.respond(prompt)
    }

    fn backend_name(&self) -> &'static str {
        "Eliza"
    }

    fn model_name(&self) -> String {
        "Weizenbaum-1966".into()
    }
}

// ── Simple REPL ───────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let eliza: Box<dyn Generator> = Box::new(MutexGenerator::new(ElizaBackend::new()));

    println!("Eliza  ·  Weizenbaum 1966  ·  eliza crate  ·  SimpleGenerator");
    println!("Type /help for info, /bye to exit.\n");

    println!(
        "Eliza > {}",
        eliza.generate("HELLO").await.unwrap_or_else(|_| "Hello.".into())
    );

    let stdin = stdin();
    let mut handle = stdin.lock();
    let mut line = String::new();

    loop {
        print!("\nYou > ");
        stdout().flush()?;
        line.clear();

        if handle.read_line(&mut line).is_err() {
            break;
        }

        let input = line.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("/bye")
            || input.eq_ignore_ascii_case("/exit")
            || input.eq_ignore_ascii_case("quit")
        {
            println!("Eliza > Goodbye.  Thank you for talking with me.");
            break;
        }

        if input.eq_ignore_ascii_case("/help") {
            println!("Eliza > I am a Rogerian psychotherapist.  Tell me about your problems.");
            println!("        Commands: /help  /bye");
            println!(
                "        Backend:   {} ({})",
                eliza.backend_name(),
                eliza.model_name()
            );
            continue;
        }

        let response = eliza.generate(&input).await?;
        println!("Eliza > {response}");
    }

    Ok(())
}
