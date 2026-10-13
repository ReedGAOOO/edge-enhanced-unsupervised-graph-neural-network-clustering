#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit[length(hit)] >= length(args)) default else args[hit[length(hit)] + 1]
}

results_root <- arg_value("--results_root", "results/urban_factor_experiments")
fig_dir <- file.path(results_root, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

city_levels <- c(
  "Bangkok", "Beijing", "Boston", "Chicago", "Johannesburg", "Madrid",
  "Melbourne", "Paris", "Shanghai", "Singapore", "Sydney", "Tokyo", "WashingtonDC"
)

read_optional <- function(path) {
  if (!file.exists(path) || file.size(path) == 0) return(tibble())
  suppressWarnings(read_csv(path, show_col_types = FALSE))
}

as_flag <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y")
}

feature_endpoint <- function(df) {
  if (nrow(df) == 0 || !"feature_set" %in% names(df)) return(tibble())
  df %>% filter(feature_set == "endpoint_edge_attr")
}

save_both <- function(plot, name, width = 12, height = 8) {
  ggsave(file.path(fig_dir, paste0(name, ".png")), plot, width = width, height = height, dpi = 220, bg = "white")
  ggsave(file.path(fig_dir, paste0(name, ".pdf")), plot, width = width, height = height, bg = "white")
}

theme_factor <- function() {
  theme_minimal(base_size = 10) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 12),
      axis.title = element_text(size = 9),
      axis.text.x = element_text(angle = 35, hjust = 1),
      legend.position = "bottom"
    )
}

placeholder <- function(title) {
  ggplot(tibble(x = 0, y = 0), aes(x, y)) +
    geom_text(aes(label = "No data available"), size = 5) +
    labs(title = title, x = NULL, y = NULL) +
    theme_void(base_size = 12)
}

label_down <- read_optional(file.path(results_root, "01_label_data", "label_downsample_runs.csv"))
label_bal <- read_optional(file.path(results_root, "01_label_data", "class_balanced_probe.csv"))
label_gran <- read_optional(file.path(results_root, "01_label_data", "label_granularity_probe.csv"))
label_sum <- read_optional(file.path(results_root, "01_label_data", "label_factor_summary.csv"))

relation_probe <- read_optional(file.path(results_root, "02_graph_projection", "relation_ablation_probe.csv"))
perm_probe <- read_optional(file.path(results_root, "02_graph_projection", "edge_attr_permutation_probe.csv"))
v4_probe <- read_optional(file.path(results_root, "02_graph_projection", "v4_construction_sweep_probe.csv"))
projection_sum <- read_optional(file.path(results_root, "02_graph_projection", "projection_factor_summary.csv"))

strat_probe <- read_optional(file.path(results_root, "03_urban_form", "stratified_edge_probe.csv"))
matched_probe <- read_optional(file.path(results_root, "03_urban_form", "matched_edge_probe.csv"))
tile_summary <- read_optional(file.path(results_root, "03_urban_form", "tile_level_form_summary.csv"))
form_sum <- read_optional(file.path(results_root, "03_urban_form", "urban_form_factor_summary.csv"))

endpoint_down <- feature_endpoint(label_down)
p11a <- if (nrow(endpoint_down) > 0) {
  endpoint_down %>%
    group_by(city_label, known_label_keep) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    mutate(city_label = factor(city_label, levels = city_levels)) %>%
    ggplot(aes(factor(known_label_keep), city_label, fill = edge_gain)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_gradient2(low = "#b2182b", mid = "white", high = "#2166ac", midpoint = 0, labels = label_number(accuracy = 0.001)) +
    labs(title = "A. Label downsampling stability", x = "Known-label keep ratio", y = NULL, fill = "AUC gain") +
    theme_factor()
} else placeholder("A. Label downsampling stability")

p11b <- if (nrow(feature_endpoint(label_bal)) > 0 && nrow(endpoint_down) > 0) {
  orig <- endpoint_down %>%
    filter(abs(known_label_keep - 1) < 1e-8) %>%
    group_by(city_label) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    mutate(condition = "original full labels")
  bal <- feature_endpoint(label_bal) %>%
    filter(balance_mode %in% c("target_balanced", "label_pair_balanced")) %>%
    group_by(city_label, balance_mode) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    mutate(condition = balance_mode) %>%
    select(city_label, edge_gain, condition)
  bind_rows(orig, bal) %>%
    mutate(city_label = factor(city_label, levels = city_levels)) %>%
    ggplot(aes(city_label, edge_gain, color = condition, group = condition)) +
    geom_hline(yintercept = 0, color = "grey75") +
    geom_point(size = 2) +
    geom_line(linewidth = 0.45) +
    labs(title = "B. Class-balance control", x = NULL, y = "AUC gain over endpoint", color = NULL) +
    theme_factor()
} else placeholder("B. Class-balance control")

p11c <- if (nrow(feature_endpoint(label_gran)) > 0) {
  feature_endpoint(label_gran) %>%
    group_by(city_label, label_mode) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    mutate(city_label = factor(city_label, levels = city_levels)) %>%
    ggplot(aes(city_label, edge_gain, fill = label_mode)) +
    geom_hline(yintercept = 0, color = "grey75") +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    labs(title = "C. Label granularity sensitivity", x = NULL, y = "AUC gain over endpoint", fill = NULL) +
    theme_factor()
} else placeholder("C. Label granularity sensitivity")

save_both((p11a / p11b / p11c) + plot_annotation(title = "Figure 11. Label/Data Controls"), "fig_11_label_data_controls", width = 12, height = 13)

endpoint_relation <- feature_endpoint(relation_probe)
p12a <- if (nrow(endpoint_relation) > 0) {
  orig <- endpoint_relation %>%
    filter(projection_condition == "original") %>%
    group_by(city_label) %>%
    summarise(original_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop")
  endpoint_relation %>%
    filter(projection_condition != "original") %>%
    group_by(city_label, projection_condition) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    left_join(orig, by = "city_label") %>%
    mutate(delta_gain = edge_gain - original_gain, city_label = factor(city_label, levels = city_levels)) %>%
    ggplot(aes(projection_condition, city_label, fill = delta_gain)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_gradient2(low = "#b2182b", mid = "white", high = "#2166ac", midpoint = 0, labels = label_number(accuracy = 0.001)) +
    labs(title = "A. Relation ablation delta", x = NULL, y = NULL, fill = "Delta AUC gain") +
    theme_factor()
} else placeholder("A. Relation ablation delta")

p12b <- if (nrow(feature_endpoint(perm_probe)) > 0) {
  perm_wide <- feature_endpoint(perm_probe) %>%
    group_by(city_label, projection_condition) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = projection_condition, values_from = edge_gain)
  if (!all(c("original", "permute_edge_attr") %in% names(perm_wide))) {
    placeholder("B. Edge-attribute permutation")
  } else {
    perm_wide %>%
      ggplot(aes(original, permute_edge_attr, label = city_label)) +
      geom_abline(slope = 1, intercept = 0, color = "grey70", linetype = 2) +
      geom_point(color = "#2166ac", size = 2.5) +
      geom_text(nudge_y = 0.0015, size = 3, check_overlap = TRUE) +
      labs(title = "B. Edge-attribute permutation", x = "Original edge gain", y = "Permuted edge-attr dataset gain") +
      theme_factor()
  }
} else placeholder("B. Edge-attribute permutation")

p12c <- if (nrow(feature_endpoint(v4_probe)) > 0) {
  feature_endpoint(v4_probe) %>%
    group_by(city_label, construction_condition) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    mutate(city_label = factor(city_label, levels = city_levels)) %>%
    ggplot(aes(construction_condition, city_label, fill = edge_gain)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_gradient2(low = "#b2182b", mid = "white", high = "#2166ac", midpoint = 0, labels = label_number(accuracy = 0.001)) +
    labs(title = "C. V4 construction sweep", x = NULL, y = NULL, fill = "AUC gain") +
    theme_factor()
} else placeholder("C. V4 construction sweep")

save_both((p12a / p12b / p12c) + plot_annotation(title = "Figure 12. Projection Controls"), "fig_12_projection_controls", width = 12, height = 13)

endpoint_strat <- feature_endpoint(strat_probe)
p13a <- if (nrow(endpoint_strat) > 0) {
  endpoint_strat %>%
    group_by(city_label, stratify_metric, stratum_label) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    group_by(city_label, stratify_metric) %>%
    summarise(gain_range = max(edge_gain, na.rm = TRUE) - min(edge_gain, na.rm = TRUE), .groups = "drop") %>%
    mutate(city_label = factor(city_label, levels = city_levels)) %>%
    ggplot(aes(stratify_metric, city_label, fill = gain_range)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_gradient(low = "white", high = "#7b3294", labels = label_number(accuracy = 0.001)) +
    labs(title = "A. Within-city stratified gain range", x = NULL, y = NULL, fill = "Range") +
    theme_factor()
} else placeholder("A. Within-city stratified gain range")

p13b <- if (nrow(feature_endpoint(matched_probe)) > 0) {
  matched_wide <- feature_endpoint(matched_probe) %>%
    group_by(city_label, match_condition) %>%
    summarise(edge_gain = mean(auc_gain_vs_endpoint, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = match_condition, values_from = edge_gain)
  if (!all(c("full_reference", "matched_distance_degree_area_l2") %in% names(matched_wide))) {
    placeholder("B. Matched-edge control")
  } else {
    matched_wide %>%
      ggplot(aes(full_reference, matched_distance_degree_area_l2, label = city_label)) +
      geom_abline(slope = 1, intercept = 0, color = "grey70", linetype = 2) +
      geom_hline(yintercept = 0, color = "grey86") +
      geom_vline(xintercept = 0, color = "grey86") +
      geom_point(color = "#008837", size = 2.5) +
      geom_text(nudge_y = 0.0015, size = 3, check_overlap = TRUE) +
      labs(title = "B. Matched-edge control", x = "Full edge gain", y = "Matched edge gain") +
      theme_factor()
  }
} else placeholder("B. Matched-edge control")

p13c <- if (nrow(tile_summary) > 0 && "edge_gain_proxy_same_label_lift" %in% names(tile_summary)) {
  tile_summary %>%
    filter(is.finite(edge_gain_proxy_same_label_lift)) %>%
    mutate(city_label = factor(city_label, levels = city_levels)) %>%
    ggplot(aes(city_label, edge_gain_proxy_same_label_lift)) +
    geom_hline(yintercept = 0, color = "grey75") +
    geom_boxplot(outlier.alpha = 0.2, fill = "#f1a340", color = "#5f3b08", linewidth = 0.3) +
    labs(title = "C. Tile-level same-label lift proxy", x = NULL, y = "Tile lift over city average") +
    theme_factor()
} else placeholder("C. Tile-level same-label lift proxy")

save_both((p13a / p13b / p13c) + plot_annotation(title = "Figure 13. Urban-Form Controls"), "fig_13_urban_form_controls", width = 12, height = 13)

all_cities <- tibble(city = unique(c(label_sum$city, projection_sum$city, form_sum$city))) %>%
  filter(!is.na(city), city != "")
decision <- all_cities %>%
  left_join(label_sum, by = "city", suffix = c("", "_label")) %>%
  left_join(projection_sum, by = "city", suffix = c("", "_projection")) %>%
  left_join(form_sum, by = "city", suffix = c("", "_form"))
for (nm in c("city_label", "city_label_projection", "city_label_form")) {
  if (!nm %in% names(decision)) decision[[nm]] <- NA_character_
}
for (nm in c(
  "label_unstable", "class_composition_sensitive", "label_task_mismatch_possible",
  "edge_attr_not_used", "projection_rule_sensitive", "spatial_geometry_confound",
  "form_mediated_edge_signal", "form_or_distance_confound", "independent_edge_semantics_supported"
)) {
  if (!nm %in% names(decision)) decision[[nm]] <- FALSE
}
decision <- decision %>%
  mutate(
    city_label = coalesce(city_label, city_label_projection, city_label_form, city),
    label_unstable = as_flag(label_unstable),
    class_composition_sensitive = as_flag(class_composition_sensitive),
    label_task_mismatch_possible = as_flag(label_task_mismatch_possible),
    edge_attr_not_used = as_flag(edge_attr_not_used),
    projection_rule_sensitive = as_flag(projection_rule_sensitive),
    spatial_geometry_confound = as_flag(spatial_geometry_confound),
    form_mediated_edge_signal = as_flag(form_mediated_edge_signal),
    form_or_distance_confound = as_flag(form_or_distance_confound),
    independent_edge_semantics_supported = as_flag(independent_edge_semantics_supported),
    decision_group = case_when(
      edge_attr_not_used | form_or_distance_confound ~ "negative control",
      projection_rule_sensitive | spatial_geometry_confound ~ "projection-risk diagnostic",
      independent_edge_semantics_supported & !label_unstable & !class_composition_sensitive ~ "main anchor",
      label_task_mismatch_possible | label_unstable | class_composition_sensitive ~ "candidate after V4",
      TRUE ~ "not recommended"
    )
  )

flag_cols <- c(
  "label_unstable",
  "class_composition_sensitive",
  "label_task_mismatch_possible",
  "edge_attr_not_used",
  "projection_rule_sensitive",
  "spatial_geometry_confound",
  "form_mediated_edge_signal",
  "form_or_distance_confound",
  "independent_edge_semantics_supported"
)

p14 <- if (nrow(decision) > 0) {
  decision_long <- decision %>%
    select(city_label, decision_group, all_of(flag_cols)) %>%
    mutate(city_label = factor(city_label, levels = city_levels)) %>%
    pivot_longer(all_of(flag_cols), names_to = "diagnostic", values_to = "flag") %>%
    mutate(
      flag = as_flag(flag),
      diagnostic = factor(diagnostic, levels = rev(flag_cols))
    )
  ggplot(decision_long, aes(diagnostic, city_label, fill = flag)) +
    geom_tile(color = "white", linewidth = 0.25) +
    geom_text(aes(label = ifelse(flag, "Y", "")), size = 3) +
    scale_fill_manual(values = c(`FALSE` = "#f0f0f0", `TRUE` = "#d6604d"), labels = c("No", "Yes")) +
    labs(title = "Figure 14. City Decision Matrix", x = NULL, y = NULL, fill = "Flag") +
    facet_grid(. ~ decision_group, scales = "free_x", space = "free_x") +
    theme_factor() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))
} else placeholder("Figure 14. City Decision Matrix")

save_both(p14, "fig_14_city_decision_matrix", width = 14, height = 8)

if (nrow(decision) > 0) {
  write_csv(decision, file.path(results_root, "city_decision_matrix.csv"))
}

cat("Urban factor experiment figures written to: ", fig_dir, "\n", sep = "")
