#![feature(internal_output_capture)]
use std::{collections::VecDeque, os::unix::io::RawFd, str};

use nix::{
    libc::_exit,
    pty::{forkpty, Winsize},
    sys::{
        select::{pselect, FdSet},
        signal::{kill, SigSet, Signal::SIGTERM},
        time::{TimeSpec, TimeValLike},
        //wait::wait,
    },
    unistd::{close, pause, pipe, read, write, ForkResult::*, Pid},
};
use termwiz::escape::{
    csi::{Cursor, CSI},
    parser::Parser as AnsiParser,
    Action, ControlCode, OneBased,
};

pub struct Pty {
    fd: RawFd,
    child: Pid,
    action_buf: VecDeque<Action>,
    child_status: ChildStatus,
}

enum ChildStatus {
    Open(RawFd),
    Closed,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub enum AnsiCmd {
    CSI(Action),
    Str(&'static str),
    CRLF,
    CursorPosition(u32, u32),
}

impl AnsiCmd {
    fn actions(&self) -> Vec<Action> {
        let mut result = Vec::new();
        match self {
            Self::CSI(action) => result.push(action.clone()),
            Self::Str(s) => {
                for c in s.chars() {
                    result.push(Action::Print(c));
                }
            }
            Self::CRLF => {
                result.push(Action::Control(ControlCode::CarriageReturn));
                result.push(Action::Control(ControlCode::LineFeed));
            }
            Self::CursorPosition(line, col) => {
                result.push(Action::CSI(CSI::Cursor(Cursor::Position {
                    line: OneBased::new(*line),
                    col: OneBased::new(*col),
                })));
            }
        }
        result
    }
}

impl Pty {
    pub fn child_is_closed(&mut self) -> bool {
        match self.child_status {
            ChildStatus::Open(fd) => {
                if select_read(fd, Some(10)) {
                    self.child_status = ChildStatus::Closed;
                    close(fd).unwrap();
                    true
                } else {
                    false
                }
            }
            ChildStatus::Closed => true,
        }
    }

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
        // Set up pipe to communicate with child process.
        let (p_parent, p_child) = pipe().unwrap();
        // Fork, and attach child process to pty.
        let pty = unsafe { forkpty(Some(&winsize), None).unwrap() };
        match pty.fork_result {
            Child => {
                close(p_parent).unwrap();
                f();
                //write(p_child, b"ok\n").unwrap();
                close(p_child).unwrap();
                pause();
                unsafe { _exit(0) }
            }
            Parent { child } => {
                close(p_child).unwrap();
                Pty {
                    fd: pty.master,
                    child,
                    action_buf: VecDeque::new(),
                    child_status: ChildStatus::Open(p_parent),
                }
            }
        }
    }

    pub fn i(&self, s: &str) {
        write(self.fd, s.as_bytes()).unwrap();
    }

    pub fn expect_sparse(&mut self, expected: &[AnsiCmd]) {
        //let expected: Vec<Vec<Action>> = expected.iter().map(|x| x.actions()).collect();
        let expected: Vec<Vec<Action>> = expected
            .iter()
            .flat_map(|x| match x {
                // Sparse match between characters of string.
                AnsiCmd::Str(s) => {
                    let mut actions = Vec::new();
                    for c in s.chars() {
                        actions.push(vec![Action::Print(c)]);
                    }
                    actions
                }
                _ => vec![x.actions()],
            })
            .collect();
        let mut matched: Vec<Vec<Action>> = Vec::new();
        for action_set in &expected[..] {
            let mut matches = Vec::new();
            let mut missed: Option<Vec<Action>> = None;
            while let Some(next) = self.next_action() {
                if next == action_set[matches.len()] {
                    matches.push(next);
                } else {
                    if missed.is_none() {
                        missed = Some(matches.clone());
                    }
                    matches.clear();
                }
                if matches.len() == action_set.len() {
                    matched.push(matches);
                    break;
                }
            }
            if let Some(missed) = missed {
                if !missed.is_empty() {
                    matched.push(missed);
                }
            }
        }
        self.action_buf.drain(..);
        assert_eq!(expected, matched);
    }

    pub fn expect_contig(&mut self, expected: &[AnsiCmd]) {
        let mut matched = Vec::new();
        let expected: Vec<Action> = expected.iter().flat_map(|x| x.actions()).collect();
        let mut missed: Option<Action> = None;
        while let Some(next) = self.next_action() {
            if next == expected[matched.len()] {
                matched.push(next);
            } else {
                //matched.clear();
                missed = Some(next);
                break;
            }
            if matched.len() == expected.len() {
                break;
            }
        }
        if let Some(missed) = missed {
            matched.push(missed);
        }
        self.action_buf.drain(..);
        assert_eq!(expected, matched);
    }

    pub fn expect(&mut self, expected: &[AnsiCmd]) {
        let mut matched = Vec::new();
        let expected: Vec<Action> = expected.iter().flat_map(|x| x.actions()).collect();
        for action in &expected[..] {
            if let Some(next) = self.next_action() {
                matched.push(next.clone());
                if action != &next {
                    break;
                }
            } else {
                break;
            }
        }
        assert_eq!(expected, matched);
    }

    /*
    pub fn test(&mut self, f: impl Fn(&dyn Fn(&str), &mut dyn FnMut(&[Action]))) {
        let fd = self.fd;
        let i = |s: &str| {
            write(fd, s.as_bytes()).unwrap();
        };
        let mut o = |expected: &[Action]| {
            let mut matched = Vec::new();
            for action in expected {
                if let Some(next) = self.next_action() {
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
    */

    fn next_action(&mut self) -> Option<Action> {
        let mut buf = [0u8; 1024];
        let mut parser = AnsiParser::new();
        loop {
            if select_read(self.fd, Some(10)) {
                let len = match read(self.fd, &mut buf) {
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
        self.action_buf.pop_front()
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        kill(self.child, SIGTERM).unwrap();
        //wait().unwrap();
        close(self.fd).unwrap();
    }
}

fn select_read(fd: RawFd, timeout: Option<i64>) -> bool {
    let mut fd_set = FdSet::new();
    fd_set.insert(fd);
    let timeout = timeout.map(TimeSpec::milliseconds);
    let sigmask = SigSet::empty();
    matches!(
        pselect(None, &mut fd_set, None, None, &timeout, &sigmask),
        Ok(1)
    )
}

#[cfg(test)]
mod tests {
    //use termwiz::escape::Action::{Control, Print};
    //use termwiz::escape::ControlCode::{CarriageReturn, LineFeed};
    use super::{AnsiCmd::*, *};

    #[test]
    fn test_async_child() {
        let mut pty = Pty::with(|| {
            print!("x");
            println!("x");
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(async move {
                println!("y");
            });
            println!("z");
        });
        pty.expect_sparse(&[Str("xx"), Str("z")]);
        pty.i("hw\n");
        pty.expect(&[Str("h")]);
        pty.expect_contig(&[Str("w"), CRLF]);
    }

    #[test]
    fn test_println() {
        let mut pty = Pty::with(|| {
            print!("x");
            println!("x");
        });
        pty.expect(&[Str("x"), Str("x"), CRLF]);
    }

    #[test]
    fn hello_world_simple() {
        let mut pty = Pty::with(|| {
            print!("oof");
            print!("Name: ");
            let mut buf = String::new();
            match std::io::stdin().read_line(&mut buf) {
                Ok(_) => println!("hello {buf}"),
                Err(err) => print!("error: {err}"),
            }
        });
        pty.expect(&[]);
    }

    #[test]
    fn hello_world_ansi() {
        let mut pty = Pty::with(|| {
            print!("Name: ");
            //std::io::stdout().lock().flush().unwrap();
            let mut buf = String::new();
            match std::io::stdin().read_line(&mut buf) {
                Ok(2) => {
                    assert_eq!(buf.len(), 2); // TODO - hmmmmmmmmmmmmmmmmmmmmmmmm
                    println!("hello {buf}");
                }
                Err(err) => print!("error: {err}"),
                _ => print!("error"),
            }
        });
        pty.i("w\n");
        pty.expect(&[Str("w"), CRLF, Str("Name: hello w"), CRLF, CRLF]);
    }

    #[test]
    fn hello_world_tui() {
        use tui::backend::CrosstermBackend;
        use tui::widgets::{Block, Borders};
        use tui::Terminal;

        let _pty = Pty::with(|| {
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
        });
        // TODO this gets way too difficult..  Needs pattern matching or `contains(&[Action])`.
        //            o(&[Action::CSI(CSI::Cursor(Cursor::Position {
        //                line: OneBased::new(1),
        //                col: OneBased::new(1),
        //            }))]);
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
    use tui::{
        backend::{Backend, CrosstermBackend},
        layout::{Constraint, Direction, Layout},
        style::{Color, Modifier, Style},
        text::{Span, Spans, Text},
        widgets::{Block, Borders, List, ListItem, Paragraph},
        Frame, Terminal,
    };
    use unicode_width::UnicodeWidthStr;

    use super::{AnsiCmd::*, *};

    #[test]
    fn test_main() {
        let mut pty = Pty::with(|| {
            main().unwrap();
        });
        pty.i("ehollo\n");
        pty.i("\x1bq");
        pty.expect(&[Str("e")]);
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
        // setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // create app and run it
        let app = App::default();
        let res = run_app(&mut terminal, app);

        // restore terminal
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
