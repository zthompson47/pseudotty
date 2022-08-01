#![feature(internal_output_capture)]
pub mod logging;

use std::collections::VecDeque;
use std::os::unix::io::RawFd;
use std::str;

use nix::libc::_exit;
use nix::pty::{forkpty, Winsize};
use nix::sys::select::{pselect, FdSet};
use nix::sys::signal::{kill, SigSet, Signal::SIGTERM};
use nix::sys::time::{TimeSpec, TimeValLike};
use nix::sys::wait::wait;
use nix::unistd::{close, pause, read, write, ForkResult::*, Pid};
use termwiz::escape::{parser::Parser as AnsiParser, Action};

pub struct Pty {
    master: i32,
    child: Pid,
    //rx_action: std::sync::mpsc::Receiver<Action>,
    action_buf: VecDeque<Action>,
}

impl Pty {
    pub fn with(f: impl Fn()) -> Self {
        // Force `--nocapture` with this unstable use of a hidden function that might disappear in
        // a future version of Rust.
        // https://github.com/rust-lang/rust/blob/master/library/std/src/io/stdio.rs#L968
        std::io::set_output_capture(None);
        // Include winsize or the child app won't have an area to write to.
        let winsize = Winsize {
            ws_row: 20,
            ws_col: 80,
            ws_xpixel: 640,
            ws_ypixel: 480,
        };
        // Fork, and attach child process to pty.
        let pty = unsafe { forkpty(Some(&winsize), None).unwrap() };
        match pty.fork_result {
            Child => {
                f();
                pause();
                unsafe { _exit(0) }
            }
            Parent { child } => {
                Pty {
                    master: pty.master,
                    child,
                    action_buf: VecDeque::new(),
                }
            }
        }
    }

    pub fn test(&mut self, f: impl Fn(&dyn Fn(&str), &mut dyn FnMut(&[Action]))) {
        let master = self.master;
        let i = |s: &str| {
            write(master, s.as_bytes()).unwrap();
        };
        let mut o = |expected: &[Action]| {
            let mut matched = Vec::new();
            for action in expected {
                if let Some(next) = self.next_action() {
                    log::info!("-->> {:?}", next);
                    matched.push(next.clone());
                    if action != &next {
                        break;
                    }
                } else {
                    break;
                }
            }
            assert_eq!(expected, matched);
        };
        f(&i, &mut o);
    }

    fn next_action(&mut self) -> Option<Action> {
        let mut buf = [0u8; 1024];
        let mut parser = AnsiParser::new();
        loop {
            if select_read(self.master, Some(10)) {
                let len = match read(self.master, &mut buf) {
                    Ok(len) => len,
                    Err(_) => return None,
                };
                if len > 0 {
                    let mut actions = VecDeque::from(parser.parse_as_vec(&buf[..len]));
                    self.action_buf.append(&mut actions);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        log::info!("{:?}", self.action_buf);
        self.action_buf.pop_front()
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        kill(self.child, SIGTERM).unwrap();
        wait().unwrap();
        close(self.master).unwrap();
    }
}

fn select_read(fd: RawFd, timeout: Option<i64>) -> bool {
    let mut fd_set = FdSet::new();
    fd_set.insert(fd);
    let timeout = timeout.map(TimeSpec::milliseconds);
    let sigmask = SigSet::empty();
    pselect(None, &mut fd_set, None, None, &timeout, &sigmask).unwrap() == 1
}

#[cfg(test)]
mod tests {
    use termwiz::escape::csi::{Cursor, CSI};
    use termwiz::escape::ControlCode::{CarriageReturn, LineFeed};
    use termwiz::escape::{
        Action::{self, Control, Print},
        OneBased,
    };
    use tui::backend::CrosstermBackend;
    use tui::widgets::{Block, Borders};
    use tui::Terminal;

    use super::*;

    #[test]
    fn test_async_child() {
        let _l = crate::logging::devlog();

        Pty::with(|| {
            println!("x");
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(async move {
                println!("y");
            });
            println!("z");
        })
        .test(|i, o| {
            i("hw\n");
            o(&[
                Print('h'),
                Print('w'),
                Control(CarriageReturn),
                Control(LineFeed),
            ]);
            o(&[
                Print('x'),
                Control(CarriageReturn),
                Control(LineFeed),
                Print('y'),
                Control(CarriageReturn),
                Control(LineFeed),
                Print('z'),
                Control(CarriageReturn),
                Control(LineFeed),
            ]);
        });
    }

    #[test]
    fn test_println() {
        let _l = crate::logging::devlog();

        Pty::with(|| {
            print!("x");
            println!("x");
        })
        .test(|_i, o| {
            o(&[
                Print('x'),
                //Print('o'),
                Print('x'),
                Control(CarriageReturn),
                Control(LineFeed),
                //Print('1'),
                //Print('2'),
            ]);
        });
    }

    #[test]
    fn hello_world_simple() {
        Pty::with(|| {
            print!("oof");
            print!("Name: ");
            let mut buf = String::new();
            match std::io::stdin().read_line(&mut buf) {
                Ok(_) => println!("hello {buf}"),
                Err(err) => print!("error: {err}"),
            }
        })
        .test(|_i, o| {
            // Nothing shows up without flushing stdout.
            o(&[]);
        });
    }

    #[test]
    fn hello_world_ansi() {
        Pty::with(|| {
            print!("Name: ");
            //std::io::stdout().lock().flush().unwrap();
            let mut buf = String::new();
            match std::io::stdin().read_line(&mut buf) {
                Ok(2) => {
                    //assert_eq!(buf.len(), 2); // TODO - hmmmmmmmmmmmmmmmmmmmmmmmm
                    println!("hello {buf}");
                }
                Err(err) => print!("error: {err}"),
                _ => print!("error"),
            }
        })
        .test(|i, o| {
            i("w\n");
            o(&[
                Print('w'),
                Control(CarriageReturn),
                Control(LineFeed),
                Print('N'),
                Print('a'),
                Print('m'),
                Print('e'),
                Print(':'),
                Print(' '),
                Print('h'),
                Print('e'),
                Print('l'),
                Print('l'),
                Print('o'),
                Print(' '),
                Print('w'),
                Control(CarriageReturn),
                Control(LineFeed),
                Control(CarriageReturn),
                Control(LineFeed),
            ]);
        });
    }

    #[test]
    fn hello_world_tui() {
        Pty::with(|| {
            let stdout = std::io::stdout();
            let backend = CrosstermBackend::new(stdout);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal
                .draw(|f| {
                    let size = f.size();
                    let block = Block::default().title("Block").borders(Borders::ALL);
                    f.render_widget(block, size);
                })
                .unwrap();
        })
        .test(|_i, o| {
            // TODO this gets way too difficult..  Needs pattern matching or `contains(&[Action])`.
            o(&[Action::CSI(CSI::Cursor(Cursor::Position {
                line: OneBased::new(1),
                col: OneBased::new(1),
            }))]);
        });
    }
}

#[cfg(test)]
mod tests_tui {
    use std::{error::Error, io};

    use crossterm::{
        event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use termwiz::escape::Action::Print;
    use tui::{
        backend::{Backend, CrosstermBackend},
        layout::{Constraint, Direction, Layout},
        style::{Color, Modifier, Style},
        text::{Span, Spans, Text},
        widgets::{Block, Borders, List, ListItem, Paragraph},
        Frame, Terminal,
    };
    use unicode_width::UnicodeWidthStr;

    use super::*;

    #[test]
    fn test_main() {
        Pty::with(|| {
            main().unwrap();
        })
        .test(|i, o| {
            i("ehollo\n");
            i("\x1bq");
            o(&[Print('e')]);
        });
    }

    enum InputMode {
        Normal,
        Editing,
    }

    /// App holds the state of the application
    struct App {
        /// Current value of the input box
        input: String,
        /// Current input mode
        input_mode: InputMode,
        /// History of recorded messages
        messages: Vec<String>,
    }

    impl Default for App {
        fn default() -> App {
            App {
                input: String::new(),
                input_mode: InputMode::Normal,
                messages: Vec::new(),
            }
        }
    }

    fn main() -> Result<(), Box<dyn Error>> {
        let _l = crate::logging::devlog();

        // setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // create app and run it
        let app = App::default();
        let res = run_app(&mut terminal, app);
        /*

        // restore terminal
        */
        disable_raw_mode()?;

        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        if let Err(err) = res {
            println!("{:?}", err)
        }

        Ok(())
    }

    fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> io::Result<()> {
        loop {
            terminal.draw(|f| ui(f, &app))?;

            if let Event::Key(key) = event::read()? {
                match app.input_mode {
                    InputMode::Normal => match key.code {
                        KeyCode::Char('e') => {
                            app.input_mode = InputMode::Editing;
                        }
                        KeyCode::Char('q') => {
                            return Ok(());
                        }
                        _ => {}
                    },
                    InputMode::Editing => match key.code {
                        KeyCode::Enter => {
                            app.messages.push(app.input.drain(..).collect());
                        }
                        KeyCode::Char(c) => {
                            app.input.push(c);
                        }
                        KeyCode::Backspace => {
                            app.input.pop();
                        }
                        KeyCode::Esc => {
                            app.input_mode = InputMode::Normal;
                        }
                        _ => {}
                    },
                }
            }
        }
    }

    fn ui<B: Backend>(f: &mut Frame<B>, app: &App) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(2)
            .constraints(
                [
                    Constraint::Length(1),
                    Constraint::Length(3),
                    Constraint::Min(1),
                ]
                .as_ref(),
            )
            .split(f.size());

        let (msg, style) = match app.input_mode {
            InputMode::Normal => (
                vec![
                    Span::raw("Press "),
                    Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" to exit, "),
                    Span::styled("e", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" to start editing."),
                ],
                Style::default().add_modifier(Modifier::RAPID_BLINK),
            ),
            InputMode::Editing => (
                vec![
                    Span::raw("Press "),
                    Span::styled("Esc", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" to stop editing, "),
                    Span::styled("Enter", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" to record the message"),
                ],
                Style::default(),
            ),
        };
        let mut text = Text::from(Spans::from(msg));
        text.patch_style(style);
        let help_message = Paragraph::new(text);
        f.render_widget(help_message, chunks[0]);

        let input = Paragraph::new(app.input.as_ref())
            .style(match app.input_mode {
                InputMode::Normal => Style::default(),
                InputMode::Editing => Style::default().fg(Color::Yellow),
            })
            .block(Block::default().borders(Borders::ALL).title("Input"));
        f.render_widget(input, chunks[1]);
        match app.input_mode {
            InputMode::Normal =>
                // Hide the cursor. `Frame` does this by default, so we don't need to do anything here
                {}

            InputMode::Editing => {
                // Make the cursor visible and ask tui-rs to put it at the specified coordinates after rendering
                f.set_cursor(
                    // Put cursor past the end of the input text
                    chunks[1].x + app.input.width() as u16 + 1,
                    // Move one line down, from the border to the input line
                    chunks[1].y + 1,
                )
            }
        }

        let messages: Vec<ListItem> = app
            .messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let content = vec![Spans::from(Span::raw(format!("{}: {}", i, m)))];
                ListItem::new(content)
            })
            .collect();
        let messages =
            List::new(messages).block(Block::default().borders(Borders::ALL).title("Messages"));
        f.render_widget(messages, chunks[2]);
    }
}

/*
pub fn with_pty(f: impl Fn()) -> Pty {
    // Force `--nocapture` with this unstable hidden function that might disappear in future Rust.
    // https://github.com/rust-lang/rust/blob/master/library/std/src/io/stdio.rs#L968
    std::io::set_output_capture(None);

    // Include winsize or the child app won't have an area to write to.
    let winsize = Winsize {
        ws_row: 20,
        ws_col: 80,
        ws_xpixel: 640,
        ws_ypixel: 480,
    };

    // Fork and attach child process to pty.
    let pty = unsafe { forkpty(Some(&winsize), None).unwrap() };
    match pty.fork_result {
        Child => {
            f();
            pause();
            unsafe { _exit(0) }
        }
        Parent { child } => {
            let (tx, rx) = std::sync::mpsc::channel();

            std::thread::spawn(move || {
                log::info!("enter thread");
                let mut buf = [0u8; 1024];
                let mut parser = AnsiParser::new();

                'out: loop {
                    log::info!("enter read loop");
                    if select_read(pty.master, None) {
                        log::info!("got select");
                        let len = match read(pty.master, &mut buf) {
                            Ok(len) => len,
                            Err(err) => {
                                log::info!("got PANIC");
                                panic!("{err}")
                            }
                        };
                        log::info!("got len {len}");
                        if len > 0 {
                            let actions = parser.parse_as_vec(&buf[..len]);
                            dbg!(&actions);
                            log::info!("got action vec len >> {}", actions.len());
                            for action in actions {
                                log::warn!(">>> {action:?}");
                                if tx.send(action).is_err() {
                                    log::warn!("+_+_+_+_+_+_+_+ tx.send IS_ERR");
                                    break 'out;
                                }
                            }
                        } else {
                            break 'out;
                        }
                    } else {
                        break 'out;
                    }
                }
            });

            Pty {
                master: pty.master,
                child,
                rx_action: rx,
            }
        }
    }
}
*/

/*
// For human readable output in tests.  Ansi commands might mess up your terminal, so beware.
pub fn assert_str_eq(left: &[u8], right: &[u8]) {
    assert_eq!(
        str::from_utf8(left).unwrap(),
        str::from_utf8(right).unwrap()
    )
}
*/

/* Pty
pub fn input(&self, b: &[u8]) {
    write(self.master, b).unwrap();
}
*/

/*
pub fn expect(&self, b: &[u8]) {
    let mut buf = Vec::from(b);
    read_exact(self.master, &mut buf);
    assert_str_eq(b, &buf);
}
*/

/*
pub fn expect_str(&self, _s: &str) {
    // TODO
    let mut parser = AnsiParser::new();
    let chunk = read_all(self.master);
    let _parsed = parser.parse_as_vec(&chunk);
}
*/

/*
pub fn expect_ansi(&self, expected: Vec<Action>) {
    let mut parser = AnsiParser::new();
    let chunk = read_all(self.master);
    let parsed = parser.parse_as_vec(&chunk);
    assert_eq!(expected, parsed);
}

pub fn expect_ansi2(&mut self, expected: Vec<Action>) {
    for action in expected {
        match self.rx_action.try_recv() {
            Ok(a) => assert_eq!(action, a),
            Err(TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
                match self.rx_action.try_recv() {
                    Ok(a) => assert_eq!(action, a),
                    _ => panic!(),
                }
            }
            Err(TryRecvError::Disconnected) => panic!(),
        }
    }
}
*/

/*
fn read_all(fd: RawFd) -> Vec<u8> {
    let mut result = Vec::new();
    let mut buf = [0u8; 1024];
    loop {
        if select_read(fd, Some(10)) {
            let len = read(fd, &mut buf).unwrap();
            if len > 0 {
                let mut v = Vec::from(&buf[0..len]);
                result.append(&mut v);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    //println!("[01]_{result:?}");
    result
}
*/

/*
// Pilfered from nix::test::read_exact.
fn read_exact(fd: RawFd, buf: &mut [u8]) -> usize {
    let mut len = 0;
    while len < buf.len() {
        // get_mut would be better than split_at_mut, but it requires nightly
        let (_, remaining) = buf.split_at_mut(len);
        if select_read(fd, Some(10)) {
            len += read(fd, remaining).unwrap();
        } else {
            panic!("could only read {}/{} bytes", buf.len(), len)
        }
    }
    len
}
*/

/* TODO Pty::new(state).with(|state| {})
trait Context {
    type State;
    fn setup(&self);
    fn teardown(&self);
    fn state(&mut self) -> &mut Self::State;
}
struct TuiState;
impl Context for TuiState {
    type State = Self;

    fn setup(&self) {
        todo!()
    }
    fn teardown(&self) {
        todo!()
    }
    fn state(&mut self) -> &mut TuiState {
        todo!()
    }
}
*/

                /*
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let mut buf = [0u8; 1024];
                    let mut parser = AnsiParser::new();
                    'out: loop {
                        if select_read(pty.master, None) {
                            let len = match read(pty.master, &mut buf) {
                                Ok(len) => len,
                                Err(_) => break 'out,
                            };
                            if len > 0 {
                                let actions = parser.parse_as_vec(&buf[..len]);
                                //dbg!(&actions);
                                for action in actions {
                                    if tx.send(action).is_err() {
                                        break 'out;
                                    }
                                }
                            } else {
                                break 'out;
                            }
                        } else {
                            break 'out;
                        }
                    }
                });
                */
