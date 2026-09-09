#!/usr/bin/env Rscript
# R adapter: stats::lm, sandwich, AER::ivreg, plm, urca.
#
# Nothing in this file was translated from the Python reference kernel. It is
# written against the R idioms for each estimator, and it recomputes every
# scalar from the definitions in the spec contract rather than trusting the
# name a package happens to give a quantity. Agreement with the reference is
# therefore evidence about the estimators, not about a shared implementation.
#
# Usage: Rscript adapters/r/run_r.R --out results/r

suppressPackageStartupMessages({
  library(jsonlite)
  library(sandwich)
  library(AER)
  library(plm)
  library(urca)
})

ADAPTER <- "r-base-sandwich-aer-plm"

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- which(args == flag)
  if (length(i) == 1 && length(args) > i) args[i + 1] else default
}
root <- get_arg("--root", ".")
out_dir <- get_arg("--out", "results/r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

param_table <- function(names_vec, beta, se, df_resid, with_p = TRUE) {
  out <- list()
  for (i in seq_along(names_vec)) {
    tstat <- as.numeric(beta[i] / se[i])
    entry <- list(
      estimate = as.numeric(beta[i]),
      se = as.numeric(se[i]),
      tstat = tstat
    )
    if (with_p) entry$pvalue <- 2 * pt(-abs(tstat), df_resid)
    out[[names_vec[i]]] <- entry
  }
  out
}

strip_prefix <- function(x) sub("^Xm", "", x)

run_ols <- function(spec, df) {
  m <- spec$model
  y <- as.numeric(df[[m$response]])
  regs <- unlist(m$regressors)
  cols <- list()
  nms <- character(0)
  if (isTRUE(m$intercept)) {
    cols[["const"]] <- rep(1, nrow(df))
    nms <- c(nms, "const")
  }
  for (r in regs) {
    cols[[r]] <- as.numeric(df[[r]])
    nms <- c(nms, r)
  }
  Xm <- do.call(cbind, cols)
  colnames(Xm) <- nms
  fit <- lm(y ~ Xm - 1)
  V <- if (identical(m$vcov, "nonrobust")) vcov(fit) else vcovHC(fit, type = m$vcov)
  beta <- coef(fit)
  se <- sqrt(diag(V))
  names(beta) <- strip_prefix(names(beta))
  n <- nrow(df)
  k <- ncol(Xm)
  df_resid <- n - k
  resid <- y - as.numeric(Xm %*% beta[nms])
  ss_resid <- sum(resid^2)
  ss_total <- if (isTRUE(m$intercept)) sum((y - mean(y))^2) else sum(y^2)
  r2 <- 1 - ss_resid / ss_total
  adj <- if (isTRUE(m$intercept)) 1 - (1 - r2) * (n - 1) / df_resid else
    1 - (1 - r2) * n / df_resid
  list(
    params = param_table(nms, beta[nms], se, df_resid),
    scalars = list(
      nobs = n, df_resid = df_resid,
      df_model = if (isTRUE(m$intercept)) k - 1 else k,
      rank = qr(Xm)$rank, sigma2 = ss_resid / df_resid,
      ss_resid = ss_resid, ss_total = ss_total, r2 = r2, adj_r2 = adj
    )
  )
}

run_iv <- function(spec, df) {
  m <- spec$model
  exog <- unlist(m$exog)
  endog <- unlist(m$endog)
  instr <- unlist(m$instruments)
  rhs <- paste(c(exog, endog), collapse = " + ")
  inst <- paste(c(exog, instr), collapse = " + ")
  form <- as.formula(sprintf("%s ~ %s | %s", m$response, rhs, inst))
  fit <- ivreg(form, data = df)
  s <- summary(fit)
  beta <- coef(fit)
  se <- s$coefficients[, 2]
  nms <- names(beta)
  nms[nms == "(Intercept)"] <- "const"
  ordered <- c(if (isTRUE(m$intercept)) "const", exog, endog)
  names(beta) <- nms
  names(se) <- nms

  y <- as.numeric(df[[m$response]])
  n <- length(y)
  k <- length(ordered)
  df_resid <- n - k
  Xs <- cbind(if (isTRUE(m$intercept)) rep(1, n) else NULL,
              as.matrix(df[, c(exog, endog), drop = FALSE]))
  resid <- y - as.numeric(Xs %*% beta[ordered])
  ss_resid <- sum(resid^2)
  ss_total <- sum((y - mean(y))^2)

  Z <- cbind(if (isTRUE(m$intercept)) rep(1, n) else NULL,
             as.matrix(df[, c(exog, instr), drop = FALSE]))
  Ze <- cbind(if (isTRUE(m$intercept)) rep(1, n) else NULL,
              as.matrix(df[, exog, drop = FALSE]))
  d <- as.numeric(df[[endog[1]]])
  rss_u <- sum(residuals(lm.fit(Z, d))^2)
  rss_r <- sum(residuals(lm.fit(Ze, d))^2)
  q <- length(instr)
  df_den <- n - ncol(Z)
  list(
    params = param_table(ordered, beta[ordered], se[ordered], df_resid),
    scalars = list(
      nobs = n, df_resid = df_resid, rank = k,
      sigma2 = ss_resid / df_resid, ss_resid = ss_resid, ss_total = ss_total,
      r2 = 1 - ss_resid / ss_total,
      n_instruments = ncol(Z), overid_df = ncol(Z) - k,
      first_stage_F = ((rss_r - rss_u) / q) / (rss_u / df_den),
      first_stage_df_num = q, first_stage_df_den = df_den
    )
  )
}

run_panel <- function(spec, df) {
  m <- spec$model
  regs <- unlist(m$regressors)
  pdf <- pdata.frame(df, index = c(m$entity, "period"))
  form <- as.formula(sprintf("%s ~ %s", m$response, paste(regs, collapse = " + ")))
  fit <- plm(form, data = pdf, model = "within", effect = "individual")
  beta <- coef(fit)[regs]
  se <- sqrt(diag(vcov(fit)))[regs]

  y <- as.numeric(df[[m$response]])
  ent <- df[[m$entity]]
  n <- length(y)
  k <- length(regs)
  n_entities <- length(unique(ent))
  df_resid <- n - n_entities - k
  demean <- function(v) v - ave(v, ent, FUN = mean)
  yd <- demean(y)
  Xd <- sapply(regs, function(r) demean(as.numeric(df[[r]])))
  resid <- yd - as.numeric(Xd %*% beta)
  ss_resid <- sum(resid^2)
  ss_within <- sum(yd^2)
  list(
    params = param_table(regs, beta, se, df_resid),
    scalars = list(
      nobs = n, n_entities = n_entities, df_resid = df_resid, rank = k,
      sigma2 = ss_resid / df_resid, ss_resid = ss_resid,
      ss_within = ss_within, r2_within = 1 - ss_resid / ss_within
    )
  )
}

run_adf <- function(spec, df) {
  m <- spec$model
  y <- as.numeric(df[[m$series]])
  lags <- as.integer(m$lags)
  trend <- m$trend
  T <- length(y)
  n_eff <- T - lags - 1
  dy <- diff(y)
  lhs <- dy[(lags + 1):(T - 1)]
  cols <- list()
  nms <- character(0)
  if (trend %in% c("c", "ct")) {
    cols[["const"]] <- rep(1, n_eff)
    nms <- c(nms, "const")
  }
  if (trend == "ct") {
    cols[["trend"]] <- as.numeric((lags + 2):T)
    nms <- c(nms, "trend")
  }
  cols[["level.lag1"]] <- y[(lags + 1):(T - 1)]
  nms <- c(nms, "level.lag1")
  if (lags > 0) {
    for (i in 1:lags) {
      cols[[paste0("diff.lag", i)]] <- dy[(lags + 1 - i):(T - 1 - i)]
      nms <- c(nms, paste0("diff.lag", i))
    }
  }
  Xm <- do.call(cbind, cols)
  colnames(Xm) <- nms
  fit <- lm(lhs ~ Xm - 1)
  beta <- coef(fit)
  se <- sqrt(diag(vcov(fit)))
  names(beta) <- strip_prefix(names(beta))
  names(se) <- names(beta)
  df_resid <- n_eff - ncol(Xm)
  resid <- lhs - as.numeric(Xm %*% beta[nms])
  ss_resid <- sum(resid^2)
  stat <- as.numeric(beta[["level.lag1"]] / se[["level.lag1"]])

  # Independent cross check against urca. The trend parameterisation differs,
  # which leaves the statistic on the lagged level unchanged because the two
  # parameterisations span the same column space together with the constant.
  ur_type <- switch(trend, "nc" = "none", "c" = "drift", "ct" = "trend")
  ur <- ur.df(y, type = ur_type, lags = lags)
  ur_stat <- as.numeric(ur@teststat[1])
  if (abs(stat - ur_stat) > 1e-8) {
    stop(sprintf("ADF cross check failed for %s: explicit %.12f vs urca %.12f",
                 spec$id, stat, ur_stat))
  }

  list(
    params = param_table(nms, beta[nms], se[nms], df_resid, with_p = FALSE),
    scalars = list(
      nobs = n_eff, df_resid = df_resid, rank = ncol(Xm),
      sigma2 = ss_resid / df_resid, ss_resid = ss_resid,
      adf_stat = stat, lags = lags
    )
  )
}

# Record what actually produced these numbers. A conformance claim without a
# version list is a claim about nothing.
pkg_version <- function(p) as.character(utils::packageVersion(p))
env_payload <- list(
  adapter = ADAPTER,
  language = "r",
  runtime = paste(R.version$major, R.version$minor, sep = "."),
  platform = R.version$platform,
  packages = list(
    jsonlite = pkg_version("jsonlite"),
    sandwich = pkg_version("sandwich"),
    AER = pkg_version("AER"),
    plm = pkg_version("plm"),
    urca = pkg_version("urca")
  )
)
writeLines(toJSON(env_payload, auto_unbox = TRUE, pretty = 2),
           file.path(out_dir, "environment.json"))

spec_files <- sort(list.files(file.path(root, "specs"), pattern = "\\.json$",
                              recursive = TRUE, full.names = TRUE))
if (length(spec_files) == 0) stop("no specs found")

for (path in spec_files) {
  spec <- fromJSON(path, simplifyVector = TRUE)
  df <- read.csv(file.path(root, spec$fixture$path), check.names = FALSE)
  res <- switch(spec$model$kind,
                "ols" = run_ols(spec, df),
                "iv2sls" = run_iv(spec, df),
                "panel_fe" = run_panel(spec, df),
                "adf" = run_adf(spec, df),
                stop(sprintf("unknown model kind %s", spec$model$kind)))
  env <- list(spec_id = spec$id, adapter = ADAPTER,
              params = res$params, scalars = lapply(res$scalars, as.numeric))
  writeLines(
    toJSON(env, auto_unbox = TRUE, digits = NA, pretty = 2),
    file.path(out_dir, paste0(spec$id, ".json"))
  )
  cat(sprintf("%s ok\n", spec$id))
}
cat(sprintf("wrote %d envelopes to %s\n", length(spec_files), out_dir))
