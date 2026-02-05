# ============================================================
# 00_run_all.R  (CAPSTONE4)
# STAT 8820 Research — Sector-wise Bayesian weekly volatility proxy modeling
# ============================================================
# -----------------------------
# A) Project root (CAPSTONE4)
# -----------------------------
Sys.setenv(CAPSTONE_DIR = "/Users/ujk/SPRING2026/CAPSTONE4")
capstone_dir <- Sys.getenv("CAPSTONE_DIR")

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

paths <- list(
  root      = capstone_dir,
  code      = file.path(capstone_dir, "code"),
  data_raw  = file.path(capstone_dir, "data", "raw"),
  data_proc = file.path(capstone_dir, "data", "processed"),
  out       = file.path(capstone_dir, "outputs"),
  fig       = file.path(capstone_dir, "outputs", "figures"),
  tab       = file.path(capstone_dir, "outputs", "tables"),
  models    = file.path(capstone_dir, "models"),
  report    = file.path(capstone_dir, "report"),
  logs      = file.path(capstone_dir, "logs")
)
invisible(lapply(paths, dir.create, recursive = TRUE, showWarnings = FALSE))

# -----------------------------
# B) Packages
# -----------------------------
pkgs <- c(
  "tidyquant","dplyr","lubridate","ggplot2",
  "readr","tibble","stringr","purrr","slider","tidyr",
  "brms","tidybayes","posterior","bayesplot"
)

need <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(need) > 0) install.packages(need, repos = "https://cloud.r-project.org")

suppressPackageStartupMessages({
  library(tidyquant); library(dplyr); library(lubridate); library(ggplot2)
  library(readr); library(tibble); library(stringr); library(purrr); library(slider); library(tidyr)
  library(brms); library(tidybayes); library(posterior); library(bayesplot)
})

# -----------------------------
# C) CmdStan setup (single, clean)
# -----------------------------
use_cmdstan <- FALSE

if (requireNamespace("cmdstanr", quietly = TRUE)) {
  suppressPackageStartupMessages(library(cmdstanr))
  cmdstan_root <- file.path(capstone_dir, "cmdstan")
  dir.create(cmdstan_root, recursive = TRUE, showWarnings = FALSE)
  
  existing <- list.dirs(cmdstan_root, full.names = TRUE, recursive = FALSE)
  existing <- existing[grepl("^cmdstan-", basename(existing))]
  
  if (length(existing) == 0) {
    cmdstanr::install_cmdstan(dir = cmdstan_root)  # installs once
    existing <- list.dirs(cmdstan_root, full.names = TRUE, recursive = FALSE)
    existing <- existing[grepl("^cmdstan-", basename(existing))]
  }
  
  if (length(existing) > 0) {
    ver <- gsub("^cmdstan-", "", basename(existing))
    pick <- existing[order(package_version(ver), decreasing = TRUE)][1]
    cmdstanr::set_cmdstan_path(pick)
    ok <- tryCatch({ cmdstanr::cmdstan_version(); TRUE }, error = function(e) FALSE)
    if (ok) use_cmdstan <- TRUE
  }
}

if (use_cmdstan) {
  options(brms.backend = "cmdstanr")
  cat("Backend: cmdstanr\n")
} else {
  if (!requireNamespace("rstan", quietly = TRUE)) {
    install.packages("rstan", repos="https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(rstan))
  options(brms.backend = "rstan")
  rstan::rstan_options(auto_write = TRUE)
  cat("Backend: rstan\n")
}

options(mc.cores = parallel::detectCores())

# -----------------------------
# D) Helpers
# -----------------------------
save_csv <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(df, path)
  cat("Saved:", path, "\n")
}

save_plot <- function(p, filename, w = 10, h = 6) {
  out <- file.path(paths$fig, filename)
  if (interactive()) print(p)
  ggsave(out, plot = p, width = w, height = h, dpi = 180)
  cat("Saved plot:", out, "\n")
  out
}

safe_read_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(readr::read_csv(path, show_col_types = FALSE), error = function(e) NULL)
}

show_scalar <- function(name, value) cat(sprintf("%-45s: %s\n", name, format(value, digits = 6)))

zscore <- function(x) as.numeric((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))

rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))

coverage95 <- function(y, lo, hi) mean(y >= lo & y <= hi, na.rm = TRUE)

# PIT for calibration: PIT = P(Yrep <= y | draws)
pit_from_draws <- function(draws_mat, y) {
  # draws_mat: iterations x N
  # y: length N
  apply(draws_mat, 2, function(col) mean(col <= y[which(draws_mat[1,] == draws_mat[1,])][1])) # defensive; replaced below
}

pit_vec <- function(draws_mat, y) {
  # proper PIT computation
  # draws_mat: ndraws x N
  apply(draws_mat, 2, function(samp) mean(samp <= y[which(samp == samp)][1])) # placeholder; replaced below
}

pit_vec <- function(draws_mat, y) {
  # draws_mat: ndraws x N
  # y: length N
  vapply(seq_len(ncol(draws_mat)), function(j) mean(draws_mat[, j] <= y[j]), numeric(1))
}

# -----------------------------
# E) Ticker universe
# -----------------------------
tickers_file <- file.path(paths$data_raw, "tickers_universe.csv")
stocks <- safe_read_csv(tickers_file)

if (is.null(stocks)) {
  cat("\nNo tickers_universe.csv found. Using ETF default universe.\n")
  stocks <- tibble(symbol = c("XLF","XLV","XLI","CARZ"),
                   sector = c("Finance","Healthcare","Industrial","Automotive"))
} else {
  cat("\nLoaded expanded ticker universe from:", tickers_file, "\n")
  stopifnot(all(c("symbol","sector") %in% names(stocks)))
  stocks <- stocks %>%
    mutate(symbol = str_trim(symbol), sector = str_trim(sector)) %>%
    distinct(symbol, .keep_all = TRUE)
}

# -----------------------------
# F) Download prices
# -----------------------------
from_date <- "2020-01-01"
to_date   <- "2024-12-31"
download_prices <- function(symbols) tq_get(symbols, get = "stock.prices", from = from_date, to = to_date)

cat("\nDownloading prices...\n")
prices_raw <- tryCatch(download_prices(stocks$symbol), error = function(e) NULL)

if (is.null(prices_raw) && any(stocks$symbol == "CARZ")) {
  cat("CARZ failed. Swapping Automotive ETF to DRIV...\n")
  stocks <- stocks %>% mutate(symbol = ifelse(symbol == "CARZ", "DRIV", symbol))
  prices_raw <- download_prices(stocks$symbol)
}
if (is.null(prices_raw) || nrow(prices_raw) == 0) stop("Price download failed.")

save_csv(prices_raw, file.path(paths$data_raw, paste0("prices_raw_", timestamp, ".csv")))

# -----------------------------
# G) Daily log returns
# -----------------------------
daily <- prices_raw %>%
  left_join(stocks, by = "symbol") %>%
  arrange(symbol, date) %>%
  group_by(symbol) %>%
  mutate(log_ret = log(adjusted / lag(adjusted))) %>%
  ungroup() %>%
  filter(!is.na(log_ret))

save_csv(daily, file.path(paths$data_raw, paste0("daily_returns_", timestamp, ".csv")))

# -----------------------------
# H) Weekly panels (WRV proxy + alternative proxy)
# Two week definitions for robustness:
# - week_start = 7 (Sunday; lubridate default)
# - week_start = 1 (Monday; ISO-ish)
# WRV proxy: sqrt(sum(r^2)) within week (not annualized)
# Alt proxy: sd(r) within week
# -----------------------------
make_weekly_panel <- function(daily_df, week_start = 7) {
  weekly <- daily_df %>%
    mutate(week = floor_date(date, "week", week_start = week_start)) %>%
    group_by(symbol, sector, week) %>%
    summarise(
      n_days = sum(!is.na(log_ret)),
      mean_ret = mean(log_ret, na.rm = TRUE),
      wrv_proxy = sqrt(sum(log_ret^2, na.rm = TRUE)),   # weekly realized volatility proxy
      log_wrv_proxy = log(wrv_proxy),
      wrv_sd = sd(log_ret, na.rm = TRUE),              # alternative proxy
      log_wrv_sd = log(pmax(wrv_sd, 1e-12)),
      .groups = "drop"
    ) %>%
    filter(is.finite(log_wrv_proxy), is.finite(log_wrv_sd))
  weekly
}

weekly_symbol_sun <- make_weekly_panel(daily, week_start = 7)
weekly_symbol_mon <- make_weekly_panel(daily, week_start = 1)

save_csv(weekly_symbol_sun, file.path(paths$data_proc, paste0("weekly_symbol_panel_SUN_", timestamp, ".csv")))
save_csv(weekly_symbol_mon, file.path(paths$data_proc, paste0("weekly_symbol_panel_MON_", timestamp, ".csv")))

# Choose primary dataset (Sunday weeks) for main modeling; use Monday as robustness later
weekly_symbol <- weekly_symbol_sun

# -----------------------------
# I) Market covariates from SPY (match week definition)
# -----------------------------
make_spy_weekly <- function(from_date, to_date, week_start = 7) {
  tq_get("SPY", get="stock.prices", from=from_date, to=to_date) %>%
    arrange(date) %>%
    mutate(mkt_log_ret = log(adjusted / lag(adjusted))) %>%
    filter(!is.na(mkt_log_ret)) %>%
    mutate(week = floor_date(date, "week", week_start = week_start)) %>%
    group_by(week) %>%
    summarise(
      mkt_ret = mean(mkt_log_ret, na.rm = TRUE),
      mkt_wrv_proxy = sqrt(sum(mkt_log_ret^2, na.rm = TRUE)),
      log_mkt_wrv_proxy = log(mkt_wrv_proxy),
      mkt_sd = sd(mkt_log_ret, na.rm = TRUE),
      log_mkt_sd = log(pmax(mkt_sd, 1e-12)),
      .groups = "drop"
    ) %>%
    filter(is.finite(log_mkt_wrv_proxy), is.finite(mkt_ret), is.finite(log_mkt_sd))
}

spy_sun <- make_spy_weekly(from_date, to_date, week_start = 7)
spy_mon <- make_spy_weekly(from_date, to_date, week_start = 1)

save_csv(spy_sun, file.path(paths$data_proc, paste0("spy_weekly_covariates_SUN_", timestamp, ".csv")))
save_csv(spy_mon, file.path(paths$data_proc, paste0("spy_weekly_covariates_MON_", timestamp, ".csv")))

spy <- spy_sun

# -----------------------------
# J) Modeling dataset (primary: log_wrv_proxy)
# -----------------------------
dat <- weekly_symbol %>%
  left_join(spy, by="week") %>%
  filter(is.finite(log_mkt_wrv_proxy), is.finite(mkt_ret)) %>%
  mutate(
    mkt_ret_z = zscore(mkt_ret),
    log_mkt_wrv_proxy_z = zscore(log_mkt_wrv_proxy)
  )

save_csv(dat, file.path(paths$data_proc, paste0("weekly_panel_dataset_MAIN_", timestamp, ".csv")))

# quick plot (main)
p_weekly <- ggplot(dat, aes(week, log_wrv_proxy, color=symbol)) +
  geom_line(alpha=0.6) +
  facet_wrap(~sector, scales="free_y") +
  theme_minimal() +
  labs(
    title="Weekly log(WRV proxy) by sector",
    subtitle="WRV proxy = sqrt(sum of daily log-return^2) within week",
    x="Week", y="log(WRV proxy)"
  )
save_plot(p_weekly, paste0("weekly_log_WRVproxy_by_sector_", timestamp, ".png"), 11, 7)

# -----------------------------
# K) Time split + blocked time folds for proper time-series evaluation
# -----------------------------
cut_date <- as.Date("2023-01-01")
train <- dat %>% filter(week < cut_date) %>% arrange(symbol, week)
test  <- dat %>% filter(week >= cut_date) %>% arrange(symbol, week)
cat("\nHoldout split — Train:", nrow(train), " Test:", nrow(test), "\n")

# Blocked time folds (few refits; much cheaper than full rolling refits)
# You can change N_FOLDS without touching the rest.
N_FOLDS <- 4

make_time_folds <- function(df, date_var="week", n_folds=4, min_train_weeks=52) {
  weeks <- sort(unique(df[[date_var]]))
  weeks <- weeks[!is.na(weeks)]
  # fold cutpoints over the last ~50% of the period
  start_idx <- which(weeks >= weeks[1] + weeks(min_train_weeks))[1]
  if (is.na(start_idx)) stop("Not enough history for folds; reduce min_train_weeks.")
  candidate <- weeks[start_idx:length(weeks)]
  cuts <- candidate[round(seq(1, length(candidate)-1, length.out=n_folds))]
  cuts <- unique(cuts)
  folds <- lapply(cuts, function(cut) {
    list(
      train_weeks = weeks[weeks < cut],
      test_weeks  = weeks[weeks >= cut & weeks < (cut + weeks(26))] # 26-week evaluation window per fold
    )
  })
  folds
}

folds <- make_time_folds(dat, n_folds = N_FOLDS, min_train_weeks = 60)
save_csv(
  tibble(
    fold = seq_along(folds),
    train_start = sapply(folds, function(f) min(f$train_weeks)),
    train_end   = sapply(folds, function(f) max(f$train_weeks)),
    test_start  = sapply(folds, function(f) min(f$test_weeks)),
    test_end    = sapply(folds, function(f) max(f$test_weeks))
  ),
  file.path(paths$tab, paste0("time_folds_", timestamp, ".csv"))
)

# -----------------------------
# L) Fit + eval helpers (Bayesian)
# -----------------------------
eval_brms_forecast <- function(model, new_df, ndraws = 800) {
  # posterior predictive (for uncertainty + PIT)
  draws <- brms::posterior_predict(model, newdata=new_df, ndraws=ndraws)
  pred_med  <- apply(draws, 2, median)
  pred_lo95 <- apply(draws, 2, quantile, probs=0.025)
  pred_hi95 <- apply(draws, 2, quantile, probs=0.975)
  
  y <- new_df$log_wrv_proxy
  pit <- pit_vec(draws, y)
  
  eval_tbl <- new_df %>%
    mutate(
      pred_med = pred_med,
      pred_lo95 = pred_lo95,
      pred_hi95 = pred_hi95,
      err = y - pred_med,
      covered95 = (y >= pred_lo95 & y <= pred_hi95),
      pit = pit
    )
  
  list(
    eval_tbl = eval_tbl,
    rmse = rmse(eval_tbl$err),
    cov95 = mean(eval_tbl$covered95, na.rm=TRUE)
  )
}

sampler_diagnostics <- function(model, tag) {
  np <- tryCatch(brms::nuts_params(model), error = function(e) NULL)
  if (is.null(np)) return(NULL)
  
  div <- mean(np$Parameter == "divergent__" & np$Value == 1)
  td  <- mean(np$Parameter == "treedepth__" & np$Value >= 12) # count hits at/above typical raised depth
  diag_tbl <- tibble(
    model = tag,
    n_divergent = sum(np$Parameter == "divergent__" & np$Value == 1),
    n_treedepth_ge12 = sum(np$Parameter == "treedepth__" & np$Value >= 12),
    n_total = nrow(np)
  )
  diag_tbl
}

plot_pit <- function(eval_tbl, title, fname) {
  p <- ggplot(eval_tbl, aes(pit)) +
    geom_histogram(bins = 20) +
    theme_minimal() +
    labs(title = title, x="PIT", y="Count")
  save_plot(p, fname, 9, 5)
}

plot_forecast_scatter <- function(eval_tbl, title, fname) {
  p <- ggplot(eval_tbl, aes(x=log_wrv_proxy, y=pred_med, color=sector)) +
    geom_point(alpha=0.4) +
    geom_abline(intercept=0, slope=1) +
    theme_minimal() +
    labs(title=title, x="Observed log(WRV proxy)", y="Pred median")
  save_plot(p, fname, 9, 6)
}

sector_metrics <- function(eval_tbl) {
  eval_tbl %>%
    group_by(sector) %>%
    summarise(
      n = n(),
      rmse = rmse(err),
      cov95 = mean(covered95, na.rm=TRUE),
      .groups="drop"
    ) %>%
    arrange(rmse)
}

# -----------------------------
# M) Baselines (ARIMA, HAR-RV) — proper volatility benchmarks
# HAR on log(WRV proxy):
# y_t = a + b1*y_{t-1} + b4*avg(y_{t-1..t-4}) + b22*avg(y_{t-1..t-22}) + e_t
# Fit per symbol; forecast one-step ahead on holdout
# -----------------------------
make_lags_har <- function(df_sym) {
  df_sym %>%
    arrange(week) %>%
    mutate(
      y = log_wrv_proxy,
      y_lag1 = lag(y, 1),
      y_bar4  = slider::slide_dbl(y, ~mean(.x, na.rm=TRUE), .before=4, .after=-1, .complete=TRUE),
      y_bar22 = slider::slide_dbl(y, ~mean(.x, na.rm=TRUE), .before=22, .after=-1, .complete=TRUE)
    )
}

fit_har <- function(train_df) {
  # simple OLS; robust enough as baseline
  stats::lm(y ~ y_lag1 + y_bar4 + y_bar22, data=train_df)
}

forecast_har_holdout <- function(df_sym, cut_date) {
  df2 <- make_lags_har(df_sym) %>% filter(is.finite(y_lag1), is.finite(y_bar4), is.finite(y_bar22))
  tr <- df2 %>% filter(week < cut_date)
  te <- df2 %>% filter(week >= cut_date)
  if (nrow(tr) < 60 || nrow(te) < 5) return(NULL)
  
  fit <- tryCatch(fit_har(tr), error=function(e) NULL)
  if (is.null(fit)) return(NULL)
  
  pred <- tryCatch(predict(fit, newdata=te, se.fit=TRUE), error=function(e) NULL)
  if (is.null(pred)) return(NULL)
  
  mu <- as.numeric(pred$fit)
  se <- as.numeric(pred$se.fit)
  te %>%
    mutate(
      pred = mu,
      se = se,
      lo95 = pred - 1.96*se,
      hi95 = pred + 1.96*se,
      err = y - pred,
      covered95 = (y >= lo95 & y <= hi95)
    )
}

forecast_arima_holdout <- function(df_sym, cut_date, order=c(1,0,0)) {
  df2 <- df_sym %>% arrange(week) %>% mutate(y = log_wrv_proxy)
  tr <- df2 %>% filter(week < cut_date)
  te <- df2 %>% filter(week >= cut_date)
  y <- tr$y
  h <- nrow(te)
  if (length(y) < 60 || h < 5) return(NULL)
  
  fit <- tryCatch(arima(y, order=order), error=function(e) NULL)
  if (is.null(fit)) return(NULL)
  
  pr <- tryCatch(predict(fit, n.ahead=h), error=function(e) NULL)
  if (is.null(pr)) return(NULL)
  
  mu <- as.numeric(pr$pred)
  se <- as.numeric(pr$se)
  te %>% mutate(
    pred = mu, se = se,
    lo95 = pred - 1.96*se, hi95 = pred + 1.96*se,
    err = y - pred,
    covered95 = (y >= lo95 & y <= hi95)
  )
}

# -----------------------------
# N) Bayesian models
# Base Student-t and Lagged Student-t with stronger sampler controls
# -----------------------------
bayes_control <- list(adapt_delta = 0.99, max_treedepth = 12)

fit_brms_student_base <- function(df) {
  brm(
    log_wrv_proxy ~ 1 + mkt_ret_z + log_mkt_wrv_proxy_z +
      (1 + mkt_ret_z + log_mkt_wrv_proxy_z | sector) +
      (1 | symbol),
    data=df,
    family=student(),
    prior=c(
      prior(normal(-3,1), class="Intercept"),
      prior(normal(0,0.5), class="b"),
      prior(student_t(3,0,0.5), class="sd"),
      prior(student_t(3,0,0.5), class="sigma"),
      prior(exponential(1), class="nu")
    ),
    chains=4, iter=4000, warmup=1000, seed=42,
    control = bayes_control
  )
}

fit_brms_student_lag <- function(df) {
  brm(
    log_wrv_proxy ~ 1 + log_wrv_proxy_lag1 + mkt_ret_z + log_mkt_wrv_proxy_z +
      (1 + log_mkt_wrv_proxy_z | sector) +
      (1 | symbol),
    data=df,
    family=student(),
    prior=c(
      prior(normal(-3,1), class="Intercept"),
      prior(normal(0,0.5), class="b"),
      prior(student_t(3,0,0.5), class="sd"),
      prior(student_t(3,0,0.5), class="sigma"),
      prior(exponential(1), class="nu")
    ),
    chains=4, iter=4000, warmup=1000, seed=42,
    control = bayes_control
  )
}

# add lag for Bayesian lag model
dat_lag <- dat %>%
  arrange(symbol, week) %>%
  group_by(symbol) %>%
  mutate(log_wrv_proxy_lag1 = lag(log_wrv_proxy, 1)) %>%
  ungroup() %>%
  filter(is.finite(log_wrv_proxy_lag1))

train_lag <- dat_lag %>% filter(week < cut_date)
test_lag  <- dat_lag %>% filter(week >= cut_date)

# -----------------------------
# O) Fit Bayesian models (single holdout)
# -----------------------------
cat("\nFitting Bayesian base Student-t (holdout split)...\n")
m_base <- fit_brms_student_base(train)
saveRDS(m_base, file.path(paths$models, paste0("brms_base_studentT_", timestamp, ".rds")))
cat("Saved model: base\n")

cat("\nFitting Bayesian lagged Student-t (holdout split)...\n")
m_lag <- fit_brms_student_lag(train_lag)
saveRDS(m_lag, file.path(paths$models, paste0("brms_lag_studentT_", timestamp, ".rds")))
cat("Saved model: lag\n")

# Diagnostics tables
diag_base <- sampler_diagnostics(m_base, "bayes_base")
diag_lag  <- sampler_diagnostics(m_lag,  "bayes_lag")
diag_all <- bind_rows(diag_base, diag_lag)
if (!is.null(diag_all)) save_csv(diag_all, file.path(paths$tab, paste0("sampler_diagnostics_", timestamp, ".csv")))

# Trace + density plots for key params (saved)
# NOTE: these can be heavy; keep to key parameters only
plot_mcmc <- function(model, pars, title, fname) {
  p <- bayesplot::mcmc_trace(as.array(model), pars = pars) + ggplot2::ggtitle(title)
  out <- file.path(paths$fig, fname)
  if (interactive()) print(p)
  ggsave(out, plot = p, width = 11, height = 7, dpi = 180)
  cat("Saved plot:", out, "\n")
}
plot_mcmc(m_base, c("b_Intercept","b_mkt_ret_z","b_log_mkt_wrv_proxy_z","sigma","nu"),
          "Trace: base Student-t", paste0("trace_base_studentT_", timestamp, ".png"))
plot_mcmc(m_lag, c("b_Intercept","b_log_wrv_proxy_lag1","b_mkt_ret_z","b_log_mkt_wrv_proxy_z","sigma","nu"),
          "Trace: lag Student-t", paste0("trace_lag_studentT_", timestamp, ".png"))

# Posterior predictive checks (saved + printed)
pp1 <- pp_check(m_base, ndraws=100) + ggtitle("pp_check: base Student-t")
save_plot(pp1, paste0("ppcheck_base_", timestamp, ".png"), 10, 6)

pp2 <- pp_check(m_lag, ndraws=100) + ggtitle("pp_check: lag Student-t")
save_plot(pp2, paste0("ppcheck_lag_", timestamp, ".png"), 10, 6)

# Holdout evaluation (Bayesian)
res_base <- eval_brms_forecast(m_base, test, ndraws=800)
res_lag  <- eval_brms_forecast(m_lag,  test_lag, ndraws=800)

save_csv(res_base$eval_tbl, file.path(paths$tab, paste0("eval_holdout_bayes_base_", timestamp, ".csv")))
save_csv(res_lag$eval_tbl,  file.path(paths$tab, paste0("eval_holdout_bayes_lag_", timestamp, ".csv")))

show_scalar("Bayes base holdout RMSE (log)", res_base$rmse)
show_scalar("Bayes base holdout 95% coverage", res_base$cov95)
show_scalar("Bayes lag holdout RMSE (log)", res_lag$rmse)
show_scalar("Bayes lag holdout 95% coverage", res_lag$cov95)

# Sector metrics (Bayesian holdout)
sec_base <- sector_metrics(res_base$eval_tbl)
sec_lag  <- sector_metrics(res_lag$eval_tbl)

save_csv(sec_base, file.path(paths$tab, paste0("sector_metrics_holdout_bayes_base_", timestamp, ".csv")))
save_csv(sec_lag,  file.path(paths$tab, paste0("sector_metrics_holdout_bayes_lag_", timestamp, ".csv")))

# Calibration PIT plots (Bayesian holdout)
plot_pit(res_base$eval_tbl, "PIT: Bayes base (holdout)", paste0("pit_bayes_base_holdout_", timestamp, ".png"))
plot_pit(res_lag$eval_tbl,  "PIT: Bayes lag (holdout)",  paste0("pit_bayes_lag_holdout_", timestamp, ".png"))

# Scatter observed vs predicted
plot_forecast_scatter(res_base$eval_tbl, "Observed vs Pred: Bayes base (holdout)",
                      paste0("scatter_bayes_base_holdout_", timestamp, ".png"))
plot_forecast_scatter(res_lag$eval_tbl, "Observed vs Pred: Bayes lag (holdout)",
                      paste0("scatter_bayes_lag_holdout_", timestamp, ".png"))

# -----------------------------
# P) Baseline evaluations (holdout)
# AR(1), ARMA(1,1), HAR-RV
# -----------------------------
by_symbol <- split(dat, dat$symbol)

ar1_eval <- map_dfr(by_symbol, ~forecast_arima_holdout(.x, cut_date, order=c(1,0,0)))
arma_eval <- map_dfr(by_symbol, ~forecast_arima_holdout(.x, cut_date, order=c(1,0,1)))
har_eval <- map_dfr(by_symbol, ~forecast_har_holdout(.x, cut_date))

save_csv(ar1_eval,  file.path(paths$tab, paste0("eval_holdout_ar1_", timestamp, ".csv")))
save_csv(arma_eval, file.path(paths$tab, paste0("eval_holdout_arma11_", timestamp, ".csv")))
save_csv(har_eval,  file.path(paths$tab, paste0("eval_holdout_har_", timestamp, ".csv")))

ar1_rmse  <- rmse(ar1_eval$err);  ar1_cov  <- mean(ar1_eval$covered95, na.rm=TRUE)
arma_rmse <- rmse(arma_eval$err); arma_cov <- mean(arma_eval$covered95, na.rm=TRUE)
har_rmse  <- rmse(har_eval$err);  har_cov  <- mean(har_eval$covered95, na.rm=TRUE)

show_scalar("AR(1) holdout RMSE (log)", ar1_rmse)
show_scalar("AR(1) holdout 95% coverage", ar1_cov)
show_scalar("ARMA(1,1) holdout RMSE (log)", arma_rmse)
show_scalar("ARMA(1,1) holdout 95% coverage", arma_cov)
show_scalar("HAR holdout RMSE (log)", har_rmse)
show_scalar("HAR holdout 95% coverage", har_cov)

# Baseline sector metrics
sec_ar1  <- ar1_eval %>% group_by(sector) %>% summarise(n=n(), rmse=rmse(err), cov95=mean(covered95,na.rm=TRUE), .groups="drop")
sec_arma <- arma_eval %>% group_by(sector) %>% summarise(n=n(), rmse=rmse(err), cov95=mean(covered95,na.rm=TRUE), .groups="drop")
sec_har  <- har_eval %>% group_by(sector) %>% summarise(n=n(), rmse=rmse(err), cov95=mean(covered95,na.rm=TRUE), .groups="drop")

save_csv(sec_ar1,  file.path(paths$tab, paste0("sector_metrics_holdout_ar1_", timestamp, ".csv")))
save_csv(sec_arma, file.path(paths$tab, paste0("sector_metrics_holdout_arma11_", timestamp, ".csv")))
save_csv(sec_har,  file.path(paths$tab, paste0("sector_metrics_holdout_har_", timestamp, ".csv")))

# -----------------------------
# Q) Proper time-series evaluation (blocked folds) for Bayes + HAR
# Refit only N_FOLDS times per model (manageable runtime)
# -----------------------------
eval_fold_models <- function(folds, dat, dat_lag) {
  fold_rows <- list()
  
  for (k in seq_along(folds)) {
    f <- folds[[k]]
    tr <- dat %>% filter(week %in% f$train_weeks)
    te <- dat %>% filter(week %in% f$test_weeks)
    
    tr_lag <- dat_lag %>% filter(week %in% f$train_weeks)
    te_lag <- dat_lag %>% filter(week %in% f$test_weeks)
    
    cat("\n[Fold", k, "] Train weeks:", min(tr$week), "to", max(tr$week),
        " | Test weeks:", min(te$week), "to", max(te$week), "\n")
    
    # Bayes base
    mb <- fit_brms_student_base(tr)
    rb <- eval_brms_forecast(mb, te, ndraws=400)  # lower draws per fold to speed up
    
    # Bayes lag
    ml <- fit_brms_student_lag(tr_lag)
    rl <- eval_brms_forecast(ml, te_lag, ndraws=400)
    
    # HAR baseline (holdout-like within fold)
    by_sym_te <- split(te, te$symbol)
    har_k <- map_dfr(by_sym_te, function(df_sym) {
      # fit HAR on fold train for that symbol
      df_full <- bind_rows(tr %>% filter(symbol==unique(df_sym$symbol)),
                           te %>% filter(symbol==unique(df_sym$symbol))) %>% arrange(week)
      forecast_har_holdout(df_full, cut_date = min(te$week))
    })
    har_k_rmse <- rmse(har_k$err)
    har_k_cov  <- mean(har_k$covered95, na.rm=TRUE)
    
    fold_rows[[k]] <- tibble(
      fold = k,
      train_end = max(tr$week),
      test_start = min(te$week),
      test_end = max(te$week),
      bayes_base_rmse = rb$rmse,
      bayes_base_cov95 = rb$cov95,
      bayes_lag_rmse = rl$rmse,
      bayes_lag_cov95 = rl$cov95,
      har_rmse = har_k_rmse,
      har_cov95 = har_k_cov
    )
    
    # save fold-level PIT plots (optional; keep small)
    plot_pit(rb$eval_tbl, paste0("PIT: Bayes base (fold ", k, ")"),
             paste0("pit_fold",k,"_bayes_base_", timestamp, ".png"))
    plot_pit(rl$eval_tbl, paste0("PIT: Bayes lag (fold ", k, ")"),
             paste0("pit_fold",k,"_bayes_lag_", timestamp, ".png"))
  }
  
  bind_rows(fold_rows)
}

fold_summary <- eval_fold_models(folds, dat, dat_lag)
save_csv(fold_summary, file.path(paths$tab, paste0("timeseries_fold_summary_", timestamp, ".csv")))

# Fold summary plot (RMSE across folds)
p_fold <- fold_summary %>%
  pivot_longer(cols = ends_with("_rmse"), names_to="model", values_to="rmse") %>%
  ggplot(aes(x=factor(fold), y=rmse, group=model, color=model)) +
  geom_line() + geom_point() +
  theme_minimal() +
  labs(title="Time-series blocked folds: RMSE by model", x="Fold", y="RMSE (log)")
save_plot(p_fold, paste0("timeseries_folds_rmse_", timestamp, ".png"), 10, 6)

# -----------------------------
# R) Sector signatures (for lag model; interpretable)
# baseline = exp(intercept_sector)
# stress multiplier for +1 SD market WRV proxy = exp(beta_stress_sector)
# -----------------------------
compute_sector_signatures <- function(model, slope_term = "log_mkt_wrv_proxy_z", prob = 0.95) {
  re_names <- names(brms::ranef(model))
  if (!("sector" %in% re_names)) {
    stop("Model has no random effects named 'sector'. Found: ", paste(re_names, collapse = ", "))
  }
  
  sector_re <- tidybayes::spread_draws(model, r_sector[sector, term]) %>%
    dplyr::rename(b_sector = r_sector) %>%
    dplyr::filter(term %in% c("(Intercept)", "Intercept", slope_term))
  
  fx <- posterior::as_draws_df(model)
  b_slope_name <- paste0("b_", slope_term)
  if (!(b_slope_name %in% names(fx))) stop("Missing fixed slope draws: ", b_slope_name)
  
  fixed_draws <- fx %>%
    dplyr::transmute(.draw = .draw, b0 = b_Intercept, bS = .data[[b_slope_name]])
  
  sig_long <- sector_re %>%
    dplyr::left_join(fixed_draws, by = ".draw") %>%
    dplyr::mutate(abs_coef = dplyr::case_when(
      term %in% c("(Intercept)", "Intercept") ~ b0 + b_sector,
      term == slope_term                      ~ bS + b_sector,
      TRUE                                    ~ NA_real_
    )) %>%
    dplyr::group_by(sector, term) %>%
    tidybayes::median_qi(abs_coef, .width = prob) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      baseline_wrv_proxy = dplyr::if_else(term %in% c("(Intercept)","Intercept"), exp(abs_coef), NA_real_),
      stress_mult_1sd    = dplyr::if_else(term == slope_term, exp(abs_coef), NA_real_)
    )
  
  sig_wide <- sig_long %>%
    dplyr::select(sector, term, abs_coef, .lower, .upper, baseline_wrv_proxy, stress_mult_1sd) %>%
    tidyr::pivot_wider(
      names_from = term,
      values_from = c(abs_coef, .lower, .upper, baseline_wrv_proxy, stress_mult_1sd),
      names_sep = "__"
    )
  
  list(long = sig_long, wide = sig_wide)
}

sig <- compute_sector_signatures(m_lag, slope_term="log_mkt_wrv_proxy_z", prob=0.95)
save_csv(sig$long, file.path(paths$tab, paste0("sector_signatures_long_", timestamp, ".csv")))
save_csv(sig$wide, file.path(paths$tab, paste0("sector_signatures_wide_", timestamp, ".csv")))

# signature plot: stress multipliers by sector (median + interval)
sig_stress <- sig$long %>% filter(term == "log_mkt_wrv_proxy_z") %>% arrange(desc(abs_coef))
p_sig <- ggplot(sig_stress, aes(x=reorder(sector, abs_coef), y=stress_mult_1sd)) +
  geom_point() +
  geom_errorbar(aes(ymin=exp(.lower), ymax=exp(.upper)), width=0.15) +
  coord_flip() +
  theme_minimal() +
  labs(
    title="Sector stress sensitivity (lag model)",
    subtitle="Multiplier in WRV proxy for +1 SD market WRV proxy (posterior median + 95% interval)",
    x="Sector", y="Stress multiplier"
  )
save_plot(p_sig, paste0("sector_stress_multiplier_", timestamp, ".png"), 10, 6)

# -----------------------------
# S) Robustness checks
# 1) Week definition Monday vs Sunday (quick compare plot)
# 2) Alternative proxy log_wrv_sd instead of log_wrv_proxy (quick model fit)
# -----------------------------
# 1) Week definition plot compare
dat_mon <- weekly_symbol_mon %>%
  left_join(spy_mon, by="week") %>%
  filter(is.finite(log_mkt_wrv_proxy), is.finite(mkt_ret)) %>%
  mutate(
    mkt_ret_z = zscore(mkt_ret),
    log_mkt_wrv_proxy_z = zscore(log_mkt_wrv_proxy)
  )

p_week_comp <- ggplot() +
  geom_line(data=dat %>% group_by(week, sector) %>% summarise(y=mean(log_wrv_proxy), .groups="drop"),
            aes(week, y, color=sector), alpha=0.7) +
  geom_line(data=dat_mon %>% group_by(week, sector) %>% summarise(y=mean(log_wrv_proxy), .groups="drop"),
            aes(week, y, color=sector), linetype="dashed", alpha=0.7) +
  theme_minimal() +
  labs(
    title="Robustness: week definition Sunday (solid) vs Monday (dashed)",
    x="Week", y="Mean log(WRV proxy) across symbols"
  )
save_plot(p_week_comp, paste0("robust_week_start_compare_", timestamp, ".png"), 11, 6)

# 2) Alternative proxy (log_wrv_sd): quick fit + holdout eval
dat_sd <- weekly_symbol %>%
  left_join(spy, by="week") %>%
  filter(is.finite(log_mkt_wrv_proxy), is.finite(mkt_ret), is.finite(log_wrv_sd)) %>%
  mutate(
    mkt_ret_z = zscore(mkt_ret),
    log_mkt_wrv_proxy_z = zscore(log_mkt_wrv_proxy)
  ) %>%
  rename(log_wrv_alt = log_wrv_sd)

train_sd <- dat_sd %>% filter(week < cut_date)
test_sd  <- dat_sd %>% filter(week >= cut_date)

cat("\nRobustness fit: Student-t on alternative proxy log(wrv_sd)...\n")
m_alt <- brm(
  log_wrv_alt ~ 1 + mkt_ret_z + log_mkt_wrv_proxy_z + (1 + log_mkt_wrv_proxy_z | sector) + (1 | symbol),
  data=train_sd,
  family=student(),
  prior=c(
    prior(normal(-3,1), class="Intercept"),
    prior(normal(0,0.5), class="b"),
    prior(student_t(3,0,0.5), class="sd"),
    prior(student_t(3,0,0.5), class="sigma"),
    prior(exponential(1), class="nu")
  ),
  chains=4, iter=3000, warmup=800, seed=42,
  control = bayes_control
)
saveRDS(m_alt, file.path(paths$models, paste0("brms_altproxy_studentT_", timestamp, ".rds")))

# evaluate alt proxy
draws_alt <- brms::posterior_predict(m_alt, newdata=test_sd, ndraws=400)
pred_med_alt <- apply(draws_alt, 2, median)
pred_lo_alt  <- apply(draws_alt, 2, quantile, 0.025)
pred_hi_alt  <- apply(draws_alt, 2, quantile, 0.975)

eval_alt <- test_sd %>%
  mutate(pred_med=pred_med_alt, pred_lo95=pred_lo_alt, pred_hi95=pred_hi_alt,
         err = log_wrv_alt - pred_med,
         covered95 = (log_wrv_alt >= pred_lo95 & log_wrv_alt <= pred_hi95))

save_csv(eval_alt, file.path(paths$tab, paste0("eval_holdout_altproxy_", timestamp, ".csv")))
show_scalar("Alt proxy holdout RMSE (log)", rmse(eval_alt$err))
show_scalar("Alt proxy holdout 95% coverage", mean(eval_alt$covered95, na.rm=TRUE))

p_alt_scatter <- ggplot(eval_alt, aes(log_wrv_alt, pred_med, color=sector)) +
  geom_point(alpha=0.4) + geom_abline(intercept=0, slope=1) +
  theme_minimal() + labs(title="Alt proxy: observed vs predicted (holdout)",
                         x="Observed log(wrv_sd)", y="Pred median")
save_plot(p_alt_scatter, paste0("scatter_altproxy_holdout_", timestamp, ".png"), 9, 6)

# -----------------------------
# T) Final comparison tables (holdout + folds)
# -----------------------------
cmp_holdout <- tibble(
  model = c("bayes_lag", "bayes_base", "har", "arma11", "ar1"),
  rmse_log = c(res_lag$rmse, res_base$rmse, har_rmse, arma_rmse, ar1_rmse),
  cov95    = c(res_lag$cov95, res_base$cov95, har_cov,  arma_cov,  ar1_cov)
) %>% arrange(rmse_log)

save_csv(cmp_holdout, file.path(paths$tab, paste0("model_compare_holdout_", timestamp, ".csv")))
print(cmp_holdout)

p_cmp <- ggplot(cmp_holdout, aes(x=reorder(model, rmse_log), y=rmse_log)) +
  geom_col() +
  coord_flip() +
  theme_minimal() +
  labs(title="Holdout comparison: RMSE (lower is better)", x="Model", y="RMSE (log)")
save_plot(p_cmp, paste0("holdout_rmse_compare_", timestamp, ".png"), 9, 5)

cat("\nDONE. All artifacts under:", capstone_dir, "\n")
cat("Key outputs:\n")
cat(" - outputs/tables/model_compare_holdout_*.csv\n")
cat(" - outputs/tables/timeseries_fold_summary_*.csv\n")
cat(" - outputs/figures/* (PIT, scatter, traces, pp_check, fold RMSE, stress multipliers)\n")

# PROJECT CODE ENDS HERE
# CODES BEYOND HERE IS USED TO GENERATE TEX FILES FOR OVERLEAF COMPILATION FOR FINAL REPORT
# ---- packages ----
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(lubridate)
  library(readr)
})

daily_df <- read_csv("/Users/ujk/SPRING2026/CAPSTONE4/data/raw/daily_returns_20260203_223246.csv",
                     show_col_types = FALSE)

# rename to what the pipeline expects
daily_df <- daily_df %>%
  rename(log_return = log_ret) %>%
  mutate(
    date = as.Date(date),
    week = floor_date(date, unit = "week", week_start = 1)  # Monday
  )

stopifnot(all(c("date","symbol","sector","log_return","week") %in% names(daily_df)))

weekly_df <- daily_df %>%
  group_by(sector, symbol, week) %>%
  summarise(
    RV = sqrt(sum(log_return^2, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(log_RV = log(RV))

p_rv <- ggplot(weekly_df, aes(x = week, y = log_RV, group = symbol, color = symbol)) +
  geom_line(alpha = 0.55, linewidth = 0.35) +
  facet_wrap(~ sector, ncol = 2, scales = "free_y") +
  labs(title = "Weekly log(RV) by sector", x = "Week", y = "log(RV)") +
  theme_bw() +
  theme(legend.position = "right")


ggsave("/Users/ujk/SPRING2026/CAPSTONE4/outputs/figures/weekly_log_RV_by_sector_20260203_223246.png",
       plot = p_rv, width = 12, height = 7, dpi = 300)


# make_tex_tables.R
# Generates LaTeX tables (*.tex) from existing CSV outputs.

library(readr)
library(dplyr)
library(knitr)

out_dir <- "tables"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

write_kable_tex <- function(df, file, caption, label, digits = 4) {
  # round numeric columns for readability
  df2 <- df %>%
    mutate(across(where(is.numeric), ~ round(.x, digits)))
  
  tex <- knitr::kable(
    df2,
    format = "latex",
    booktabs = TRUE,
    caption = caption,
    label = label,
    longtable = FALSE
  )
  
  writeLines(tex, file)
  message("Wrote: ", file)
}

# ----  ----
fold_summary_path   <- "/Users/ujk/SPRING2026/CAPSTONE4/outputs/tables/timeseries_fold_summary_20260203_223246.csv"
model_compare_path  <- "/Users/ujk/SPRING2026/CAPSTONE4/outputs/tables/model_compare_holdout_20260203_223246.csv"
sampler_diag_path   <- "/Users/ujk/SPRING2026/CAPSTONE4/outputs/tables/sampler_diagnostics_20260203_223246.csv"

# Optional sector metrics 
sector_base_path <- "/Users/ujk/SPRING2026/CAPSTONE4/outputs/tables/sector_metrics_holdout_bayes_base_20260203_223246.csv"
sector_lag_path  <- "/Users/ujk/SPRING2026/CAPSTONE4/outputs/tables/sector_metrics_holdout_bayes_lag_20260203_223246.csv"
sector_har_path  <- "/Users/ujk/SPRING2026/CAPSTONE4/outputs/tables/sector_metrics_holdout_har_20260203_223246.csv"

fold_df  <- read_csv(fold_summary_path, show_col_types = FALSE)
mc_df    <- read_csv(model_compare_path, show_col_types = FALSE)
sd_df    <- read_csv(sampler_diag_path, show_col_types = FALSE)



fold_df_small <- fold_df

write_kable_tex(
  fold_df_small,
  file = file.path(out_dir, "fold_results.tex"),
  caption = "Time-series cross-validation fold summary.",
  label = "tab:fold_results"
)

write_kable_tex(
  mc_df,
  file = file.path(out_dir, "model_compare_holdout.tex"),
  caption = "Holdout performance comparison across models.",
  label = "tab:model_compare_holdout"
)

write_kable_tex(
  sd_df,
  file = file.path(out_dir, "sampler_diagnostics.tex"),
  caption = "Sampler diagnostics for Bayesian models.",
  label = "tab:sampler_diagnostics"
)

# Combine sector metrics into one table 
sector_tables <- list()
if (file.exists(sector_base_path)) sector_tables$bayes_base <- read_csv(sector_base_path, show_col_types = FALSE)
if (file.exists(sector_lag_path))  sector_tables$bayes_lag  <- read_csv(sector_lag_path,  show_col_types = FALSE)
if (file.exists(sector_har_path))  sector_tables$har        <- read_csv(sector_har_path,  show_col_types = FALSE)

if (length(sector_tables) > 0) {
  sector_df <- bind_rows(sector_tables, .id = "model")
  write_kable_tex(
    sector_df,
    file = file.path(out_dir, "sector_metrics_holdout.tex"),
    caption = "Sector-level holdout metrics by model.",
    label = "tab:sector_metrics_holdout"
  )
}

