//! CLI entry point for the `unrust-it` 3D multi-arm visualizer.
//!
//! Usage:
//! - `cargo run -p unrust-it-visualizer -- --export-html visualizer.html`
//! - `cargo run -p unrust-it-visualizer -- --export-json scenarios.json`
//! - `cargo run -p unrust-it-visualizer -- --serve 127.0.0.1:8080`

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use unrust_it_visualizer::{generate_scenarios, render_visualizer_html, scenarios_to_json};

fn main() {
    let args: Vec<String> = env::args().collect();
    let scenarios = generate_scenarios();

    let mut i = 1;
    let mut exported = false;
    let mut serve_addr: Option<String> = None;

    while i < args.len() {
        match args[i].as_str() {
            "--export-html" if i + 1 < args.len() => {
                let path = &args[i + 1];
                let html = render_visualizer_html(&scenarios);
                fs::write(path, html).expect("failed to write visualizer HTML");
                println!("Exported interactive 3D visualizer HTML to {path}");
                exported = true;
                i += 2;
            }
            "--export-json" if i + 1 < args.len() => {
                let path = &args[i + 1];
                let json = scenarios_to_json(&scenarios);
                fs::write(path, json).expect("failed to write scenarios JSON");
                println!("Exported multi-arm scenarios JSON to {path}");
                exported = true;
                i += 2;
            }
            "--serve" => {
                let addr = if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 2;
                    args[i - 1].clone()
                } else {
                    i += 1;
                    "127.0.0.1:8080".to_string()
                };
                serve_addr = Some(addr);
            }
            _ => {
                i += 1;
            }
        }
    }

    if let Some(addr) = serve_addr {
        let html = render_visualizer_html(&scenarios);
        let json = scenarios_to_json(&scenarios);
        let listener = TcpListener::bind(&addr).expect("failed to bind HTTP listener");
        println!("unrust-it 3D visualizer listening on http://{addr}");
        for mut stream in listener.incoming().flatten() {
            let mut req_buf = [0u8; 2048];
            let n = stream.read(&mut req_buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&req_buf[..n]);
            let (content_type, body) = if req.starts_with("GET /api/scenarios") {
                ("application/json; charset=utf-8", json.as_bytes())
            } else {
                ("text/html; charset=utf-8", html.as_bytes())
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body);
        }
    } else if !exported {
        println!(
            "Generated {} multi-arm scenarios ({} frames each). Pass `--export-html <path>` or `--serve 127.0.0.1:8080`.",
            scenarios.len(),
            scenarios[0].frames.len()
        );
    }
}
