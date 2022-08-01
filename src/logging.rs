use std::env;
use std::fmt::Error;
use std::path::Path;

use crossterm::style::Stylize;

use tracing::subscriber::Subscriber;
use tracing::{Event, Level};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling;
use tracing_log::NormalizeEvent;
use tracing_subscriber::{
    fmt::{format::Writer, FmtContext, FormatEvent, FormatFields},
    registry::LookupSpan,
};

pub fn devlog() -> Option<WorkerGuard> {
    dev_log()
}

pub fn dev_log() -> Option<WorkerGuard> {
    let log_dir = match env::var("Z_DEV_DIR") {
        Ok(dir) => Path::new(&dir).to_path_buf(),
        Err(_) => Path::new(".").to_path_buf(),
    };
    let default_level = Level::INFO;
    let file_appender = rolling::never(log_dir, String::from("log.txt"));
    let (log_writer, guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_max_level(match env::var("RUST_LOG") {
            Ok(level) => match level.as_str() {
                "info" | "INFO" => Level::INFO,
                "warn" | "WARN" => Level::WARN,
                "error" | "ERROR" => Level::ERROR,
                "debug" | "DEBUG" => Level::DEBUG,
                "trace" | "TRACE" => Level::TRACE,
                _ => default_level,
            },
            _ => default_level,
        })
        .with_writer(log_writer)
        .event_format(SimpleFmt)
        .try_init()
        .ok();

    Some(guard)
}

struct SimpleFmt;

impl<S, N> FormatEvent<S, N> for SimpleFmt
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> Result<(), Error> {
        let time_now = chrono::Local::now();
        let time_now = time_now.format("%b %d %I:%M:%S%.6f %p").to_string();
        // Get line numbers from log crate events
        let normalized_meta = event.normalized_metadata();
        let meta = normalized_meta.as_ref().unwrap_or_else(|| event.metadata());
        let message = format!(
            "{} {} {}{}{} ",
            time_now.grey(),
            meta.level().to_string().blue(),
            //meta.file().unwrap_or("").to_string().yellow(),
            shortpath(meta.file().unwrap_or("")).yellow(),
            String::from(":").yellow(),
            meta.line().unwrap_or(0).to_string().yellow(),
        );

        write!(writer, "{}", message).unwrap();
        // Write actual log message with newline
        ctx.format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

fn shortpath(path: &str) -> String {
    let mut result = String::new();
    let mut tmp = String::new();
    for c in path.chars() {
        if c == '/' {
            if !tmp.is_empty() {
                result.push_str(&tmp[..1]);
                tmp.clear();
            }
            result.push('/');
        } else {
            tmp.push_str(String::from(c).as_str());
        }
    }
    result.push_str(tmp.as_str());
    result
}

/*
#[cfg(test)]
mod tests {
    use std::{env, fs, io::prelude::*, io::BufReader};

    use log;
    use tempfile::tempdir;
    use tracing;

    use super::*;

    #[test]
    fn generate_log_records() {
        // Use tempdir for log files
        let dir = tempdir().unwrap().into_path();
        env::set_var("TAPEDECK_DEV_DIR", &dir);
        let _logging = dev_log();

        // Try both logging crates
        log::info!("test log INFO");
        tracing::debug!("test tracing DEBUG");

        // Confirm log file created
        //let dir = dir.join("appname-test");
        //let dir = dir.join("log");
        assert!(dir.is_dir());
        let log_file = dir.join("log");
        assert!(log_file.is_file());

        // Drop logging guard to flush logs.
        // TODO test it??  failure below was intermittent
        //   hmm still dropping records after this drop..
        drop(_logging);

        // Check log records
        let file = fs::File::open(&log_file).unwrap();
        let buf_reader = BufReader::new(file);
        let log_records: Vec<String> = buf_reader.lines().map(|x| x.unwrap()).collect();
        assert!(log_records.len() >= 2);
        let idx = log_records.len() - 2;

        if !log_records[idx].contains("test log INFO") {
            // TODO - Log records are occasionally dropped... ??
            println!("{}", fs::read_to_string(&log_file).unwrap());
        }

        assert!(log_records[idx].contains("test log INFO"));
        assert!(log_records[idx + 1].ends_with("test tracing DEBUG"));
    }
}
*/
