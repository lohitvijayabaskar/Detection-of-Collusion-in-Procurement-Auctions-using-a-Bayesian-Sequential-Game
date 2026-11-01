"""Knowledge priors shared by the monotone and tail-score designs.

``TAIL_DIRECTIONS`` gives the direction in which collusion moves each feature (+1 up, -1 down).
It is fixed from the economics of each mechanism family and never fitted on labels. Only
scale-free identity and participation features appear, so price level and cost spread do not
enter.
"""

TAIL_DIRECTIONS = {
    "dep_pair_corr_pos_t": +1,
    "dep_close_pair_t": +1,
    "dep_near_tie_pair_t": +1,
    "dep_firm_sd_min_t": +1,
    "dep_cover_sd_min_t": +1,
    "dep_win_loss_corr": +1,
    "dep_cobid_min": +1,
    "part_group": +1,
    "part_rate": -1,
}
TAIL_MARGIN = 2.0
