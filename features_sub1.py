from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from collections import Counter

species = {
    'alakazam': {'hp': 55, 'atk': 50, 'def': 45, 'spa': 135, 'spd': 135, 'spe': 120},
    'articuno': {'hp': 90, 'atk': 85, 'def': 100, 'spa': 125, 'spd': 125, 'spe': 85},
    'chansey': {'hp': 250,'atk': 5, 'def': 5,  'spa': 105, 'spd': 105, 'spe': 50},
    'charizard':{'hp': 78, 'atk': 84, 'def': 78, 'spa': 85,  'spd': 85,  'spe': 100},
    'cloyster': {'hp': 50, 'atk': 95, 'def': 180,'spa': 85,  'spd': 85,  'spe': 70},
    'dragonite':{'hp': 91, 'atk': 134,'def': 95, 'spa': 100, 'spd': 100, 'spe': 80},
    'exeggutor':{'hp': 95, 'atk': 95, 'def': 85, 'spa': 125, 'spd': 125, 'spe': 55},
    'gengar':   {'hp': 60, 'atk': 65, 'def': 60, 'spa': 130, 'spd': 130, 'spe': 110},
    'golem':    {'hp': 80, 'atk': 110,'def': 130,'spa': 55,  'spd': 55,  'spe': 45},
    'jolteon':  {'hp': 65, 'atk': 65, 'def': 60, 'spa': 110, 'spd': 110, 'spe': 130},
    'jynx':     {'hp': 65, 'atk': 50, 'def': 35, 'spa': 95,  'spd': 95,  'spe': 95},
    'lapras':   {'hp': 130,'atk': 85, 'def': 80, 'spa': 95,  'spd': 95,  'spe': 60},
    'persian':  {'hp': 65, 'atk': 70, 'def': 60, 'spa': 65,  'spd': 65,  'spe': 115},
    'rhydon':   {'hp': 105,'atk': 130,'def': 120,'spa': 45,  'spd': 45,  'spe': 40},
    'slowbro':  {'hp': 95, 'atk': 75, 'def': 110,'spa': 80,  'spd': 80,  'spe': 30},
    'snorlax':  {'hp': 160,'atk': 110,'def': 65, 'spa': 65,  'spd': 65,  'spe': 30},
    'starmie':  {'hp': 60, 'atk': 75, 'def': 85, 'spa': 100, 'spd': 100, 'spe': 115},
    'tauros':   {'hp': 75, 'atk': 100,'def': 95, 'spa': 70,  'spd': 70,  'spe': 110},
    'victreebel':{'hp': 80,'atk': 105,'def': 65, 'spa': 100, 'spd': 100, 'spe': 70},
    'zapdos':   {'hp': 90, 'atk': 90, 'def': 85, 'spa': 125, 'spd': 125, 'spe': 100},
}

types = {
    "alakazam":["notype","psychic"], "articuno":["flying","ice"], "chansey":["normal","notype"],
    "charizard":["fire","flying"], "cloyster":["ice","water"], "dragonite":["dragon","flying"],
    "exeggutor":["grass","psychic"], "gengar":["ghost","poison"], "golem":["ground","rock"],
    "jolteon":["electric","notype"], "jynx":["ice","psychic"], "lapras":["ice","water"],
    "persian":["normal","notype"], "rhydon":["ground","rock"], "slowbro":["psychic","water"],
    "snorlax":["normal","notype"], "starmie":["psychic","water"], "tauros":["normal","notype"],
    "victreebel":["grass","poison"], "zapdos":["electric","flying"]
}

effectiveness = {
    'normal': {'rock':0.5,'ghost':0,'notype':1},
    'fire':{'grass':2,'ice':2,'bug':2,'rock':0.5,'fire':0.5,'water':0.5,'dragon':0.5},
    'water':{'fire':2,'rock':2,'ground':2,'water':0.5,'grass':0.5,'dragon':0.5},
    'electric':{'water':2,'flying':2,'ground':0,'electric':0.5,'grass':0.5,'dragon':0.5},
    'grass':{'water':2,'rock':2,'ground':2,'fire':0.5,'grass':0.5,'poison':0.5,'flying':0.5,'dragon':0.5},
    'ice':{'grass':2,'ground':2,'flying':2,'dragon':2,'fire':0.5,'ice':0.5,'water':0.5},
    'poison':{'grass':2,'poison':0.5,'ground':0.5,'rock':0.5,'ghost':0.5},
    'ground':{'fire':2,'electric':2,'poison':2,'rock':2,'grass':0.5,'flying':0},
    'flying':{'grass':2,'fighting':2,'bug':2,'rock':0.5,'electric':0.5},
    'psychic':{'poison':2,'fighting':2,'psychic':0.5},
    'bug':{'grass':2,'psychic':2,'poison':0.5,'fire':0.5,'flying':0.5},
    'rock':{'fire':2,'ice':2,'flying':2,'bug':2,'ground':0.5},
    'ghost':{'ghost':2,'psychic':0},
    'dragon':{'dragon':2},
    'notype':{}
}

MAP_STATUS = {'nostatus':0,'par':1,'brn':1,'psn':1,'tox':2,'frz':2,'slp':3}


def type_match(tp1, tp2):
    v = effectiveness.get(tp1, {}).get(tp2, 1)
    if v == 0:
        return -2
    if v < 1:
        return -1
    if v > 1:
        return 1
    return 0


def _safe_div(a, b):
    return float(a) / float(b) if float(b) != 0 else 0.0


def _entropy(counter: Counter):
    tot = float(sum(counter.values()))
    if tot <= 0:
        return 0.0
    p = [c / tot for c in counter.values()]
    return float(-sum(pi * np.log(max(pi, 1e-12)) for pi in p))


def _run_lengths(seq):
    if not seq:
        return []
    runs, cur, cnt = [], seq[0], 1
    for s in seq[1:]:
        if s == cur:
            cnt += 1
        else:
            runs.append(cnt)
            cur, cnt = s, 1
    runs.append(cnt)
    return runs


def create_features(data: list[dict]) -> pd.DataFrame:
    out = []
    for battle in tqdm(data, desc="Extracting features"):
        f = {}
        p1team = battle.get("p1_team_details") or []
        p1_mean_def = np.mean([int(x.get("base_def", 0)) for x in p1team]) if p1team else 0.0
        p1_mean_atk = np.mean([int(x.get("base_atk", 0)) for x in p1team]) if p1team else 0.0
        p1_mean_spe = np.mean([int(x.get("base_spe", 0)) for x in p1team]) if p1team else 0.0
        f["p1_mean_def"] = float(p1_mean_def)
        tl = battle.get("battle_timeline") or []
        den = len(tl)
        lead_p1 = str(tl[0]["p1_pokemon_state"]["name"]).lower()
        lead_p2 = str(tl[0]["p2_pokemon_state"]["name"]).lower()
        def_p1 = species.get(lead_p1, {}).get("def", 0)
        def_p2 = species.get(lead_p2, {}).get("def", 0)
        f["lead_def_edge"] = float(def_p2 - def_p1)
        lt_p1 = [t for t in types.get(lead_p1, ["notype", "notype"]) if t]
        lt_p2 = [t for t in types.get(lead_p2, ["notype", "notype"]) if t]
        lead_type_edge = 0
        for t2 in lt_p2:
            for t1 in lt_p1:
                lead_type_edge += type_match(t2, t1)
                lead_type_edge -= type_match(t1, t2)
        f["lead_type_edge"] = int(lead_type_edge)
        spe_p1 = species.get(lead_p1, {}).get("spe", 0)
        spe_p2 = species.get(lead_p2, {}).get("spe", 0)
        lead_speed_edge = spe_p2 - spe_p1
        p1_hp = [float(t["p1_pokemon_state"]["hp_pct"]) for t in tl]
        p2_hp = [float(t["p2_pokemon_state"]["hp_pct"]) for t in tl]
        p1_stat_raw = [t["p1_pokemon_state"].get("status", "nostatus") for t in tl]
        p2_stat_raw = [t["p2_pokemon_state"].get("status", "nostatus") for t in tl]
        p1_moves = [t.get("p1_move_details") or {} for t in tl]
        p2_moves = [t.get("p2_move_details") or {} for t in tl]
        p1_active = [str(t["p1_pokemon_state"]["name"]).lower() for t in tl]
        p2_active = [str(t["p2_pokemon_state"]["name"]).lower() for t in tl]
        p1_switches = sum(1 for a, b in zip(p1_active, p1_active[1:]) if a != b)
        p2_switches = sum(1 for a, b in zip(p2_active, p2_active[1:]) if a != b)
        last_sw_p1 = 1
        c = 1
        for a, b in zip(p1_active, p1_active[1:]):
            c += 1
            if a != b:
                last_sw_p1 = c
        f["last_switch_turn_p1"] = int(last_sw_p1)
        rl1 = _run_lengths(p1_active)
        rl2 = _run_lengths(p2_active)
        f["p2_run_len_mean"] = float(np.mean(rl2)) if rl2 else 0.0
        f["run_len_mean_diff"] = float(
            (np.mean(rl2) if rl2 else 0.0) - (np.mean(rl1) if rl1 else 0.0)
        )
        E = max(1, den // 3)
        M_end = max(E * 2, min(den, E * 2))
        p1_loss = [max(0.0, p1_hp[i - 1] - p1_hp[i]) for i in range(1, den)]
        p2_loss = [max(0.0, p2_hp[i - 1] - p2_hp[i]) for i in range(1, den)]
        f["p2_early_damage"] = float(sum(p2_loss[:E]))
        f["p2_late_damage"] = float(sum(p2_loss[-E:]))
        f["p1_mid_damage"] = float(sum(p1_loss[E:M_end]))
        f["early_damage_gap"] = float(sum(p2_loss[:E]) - sum(p1_loss[:E]))
        p1_heal = [max(0.0, p1_hp[i] - p1_hp[i - 1]) for i in range(1, den)]
        p2_heal = [max(0.0, p2_hp[i] - p2_hp[i - 1]) for i in range(1, den)]
        f["heal_mid_diff"] = float(sum(p2_heal[E:M_end]) - sum(p1_heal[E:M_end]))
        f["heal_late_diff"] = float(sum(p2_heal[-E:]) - sum(p1_heal[-E:]))
        hp_gap = np.array(p2_hp) - np.array(p1_hp)
        f["hp_gap_early"] = float(np.mean(hp_gap[:E])) if len(hp_gap) > 0 else 0.0
        f["hp_gap_mid"] = float(np.mean(hp_gap[E:M_end])) if len(hp_gap) > E else 0.0
        f["hp_gap_var"] = float(np.var(hp_gap)) if len(hp_gap) > 1 else 0.0
        signs = np.sign(hp_gap)
        f["hp_gap_sign_flips"] = int(
            sum(1 for a, b in zip(signs, signs[1:]) if a != 0 and b != 0 and a != b)
        )

        def _fb(prev_seq):
            return next(
                (i + 1 for i, (a, b) in enumerate(zip(prev_seq, prev_seq[1:]))
                 if a != 'fnt' and b == 'fnt'),
                None
            )

        fb_p1 = _fb(p2_stat_raw)
        fb_p2 = _fb(p1_stat_raw)
        firsts = [x for x in (fb_p1, fb_p2) if x is not None]
        fb_turn = min(firsts) if firsts else None
        f["first_blood_happened"] = int(fb_turn is not None)
        f["first_blood_side"] = (
            1 if (fb_p1 is not None and (fb_p2 is None or fb_p1 <= fb_p2))
            else (-1 if fb_p2 is not None else 0)
        )
        f["lead_type_fb_agree"] = int(
            f["first_blood_side"] != 0 and np.sign(lead_type_edge) == f["first_blood_side"]
        )
        f["lead_speed_fb_agree"] = int(
            f["first_blood_side"] != 0 and np.sign(lead_speed_edge) == f["first_blood_side"]
        )
        p1_series = [MAP_STATUS.get(s, 0) for s in p1_stat_raw]
        p2_series = [MAP_STATUS.get(s, 0) for s in p2_stat_raw]
        p1_par = sum(1 for s in p1_stat_raw if s == 'par')
        p2_par = sum(1 for s in p2_stat_raw if s == 'par')
        p1_frz = sum(1 for s in p1_stat_raw if s == 'frz')
        p1_brn = sum(1 for s in p1_stat_raw if s == 'brn')
        p2_brn = sum(1 for s in p2_stat_raw if s == 'brn')
        p1_psx = sum(1 for s in p1_stat_raw if s in {'psn', 'tox'})
        p2_psx = sum(1 for s in p2_stat_raw if s in {'psn', 'tox'})
        p2_slp = sum(1 for s in p2_stat_raw if s == 'slp')
        f["p1_turns_par"] = int(p1_par)
        f["p1_turns_frz"] = int(p1_frz)
        f["p1_turns_brn"] = int(p1_brn)
        f["p1_turns_psn_tox"] = int(p1_psx)
        f["p2_turns_brn"] = int(p2_brn)
        f["p2_turns_psn_tox"] = int(p2_psx)
        f["p2_turns_slp"] = int(p2_slp)
        f["par_turns_diff"] = int(p1_par - p2_par)
        f["severe2_turns_diff"] = int(
            (p1_frz + sum(1 for s in p1_stat_raw if s == 'slp')) -
            (p2_slp + sum(1 for s in p2_stat_raw if s == 'frz'))
        )
        severe_mask = [(a >= 2 or b >= 2) for a, b in zip(p1_series, p2_series)]
        f["severe_status_share"] = _safe_div(
            sum(1 for x in severe_mask if x), den
        )
        f["severe_status_early_share"] = _safe_div(
            sum(1 for x in severe_mask[:E] if x), max(1, E)
        )
        p1_seen = set(p1_active)
        p2_seen = set(p2_active)
        f["revealed_count_diff"] = int(len(p1_seen) - len(p2_seen))
        f["p2_switch_early_share"] = _safe_div(
            sum(1 for a, b in zip(p2_active[:E], p2_active[1:E]) if a != b),
            den
        )
        c1 = Counter(p1_active)
        c2 = Counter(p2_active)
        p1_ent = _entropy(c1)
        p2_ent = _entropy(c2)
        f["active_entropy_diff"] = float(p1_ent - p2_ent)

        def _p1_faster(i):
            n1 = p1_active[i]
            n2 = p2_active[i]
            return species.get(n1, {}).get("spe", 0) >= species.get(n2, {}).get("spe", 0)

        p1_fast = sum(1 for i in range(den) if _p1_faster(i))
        f["initiative_early_diff"] = (
            _safe_div(sum(1 for i in range(min(E, den)) if _p1_faster(i)), max(1, E))
            - _safe_div(sum(1 for i in range(min(E, den)) if not _p1_faster(i)), max(1, E))
        )
        f["initiative_late_diff"] = (
            _safe_div(sum(1 for i in range(max(0, den - E), den) if _p1_faster(i)), max(1, E))
            - _safe_div(sum(1 for i in range(max(0, den - E), den) if not _p1_faster(i)), max(1, E))
        )

        def _wmean(counter, key):
            vals = [(species[n][key], w) for n, w in counter.items() if n in species]
            return float(sum(v * w for v, w in vals)) / float(sum(w for _, w in vals)) if vals else 0.0

        p1_used_mean_spe = _wmean(c1, 'spe')
        p2_used_mean_spe = _wmean(c2, 'spe')
        p2_used_mean_atk = _wmean(c2, 'atk')
        f["p1_used_mean_spe"] = float(p1_used_mean_spe)
        f["used_mean_spe_diff"] = float(p1_used_mean_spe - p2_used_mean_spe)
        f["p2_used_count"] = int(len(c2))
        f["atk_edge_used"] = float(p2_used_mean_atk - p1_mean_atk)
        last_hp_p1, last_hp_p2, last_st_p1, last_st_p2 = {}, {}, {}, {}
        for t in tl:
            n1 = str(t["p1_pokemon_state"]["name"]).lower()
            n2 = str(t["p2_pokemon_state"]["name"]).lower()
            last_hp_p1[n1] = float(t["p1_pokemon_state"]["hp_pct"])
            last_hp_p2[n2] = float(t["p2_pokemon_state"]["hp_pct"])
            last_st_p1[n1] = t["p1_pokemon_state"].get("status", "nostatus")
            last_st_p2[n2] = t["p2_pokemon_state"].get("status", "nostatus")
        p1_alive_final = sum(1 for v in last_hp_p1.values() if v > 0)
        p2_alive_final = sum(1 for v in last_hp_p2.values() if v > 0)
        f["alive_diff_final"] = int(p1_alive_final - p2_alive_final)
        mean_hp_p1 = float(np.mean(list(last_hp_p1.values()))) if last_hp_p1 else 0.0
        mean_hp_p2 = float(np.mean(list(last_hp_p2.values()))) if last_hp_p2 else 0.0
        f["hp_edge_final"] = float(mean_hp_p2 - mean_hp_p1)
        f["mean_remaining_hp_p2"] = float(mean_hp_p2)
        p1_status_mean_final = float(np.mean([MAP_STATUS.get(s, 0) for s in last_st_p1.values()])) if last_st_p1 else 0.0
        p2_status_mean_final = float(np.mean([MAP_STATUS.get(s, 0) for s in last_st_p2.values()])) if last_st_p2 else 0.0
        f["p1_status_mean_final"] = float(p1_status_mean_final)
        f["status_severity_gap_final"] = float(p2_status_mean_final - p1_status_mean_final)

        def _avg_edge(side_key, opp_key):
            vals = []
            for t in tl:
                md = t.get(side_key) or {}
                if not md.get("name"):
                    continue
                mt = str(md.get("type", "")).lower()
                if not mt:
                    continue
                oppn = str(t[opp_key]["name"]).lower()
                opp_types = [tp for tp in types.get(oppn, ["notype", "notype"]) if tp != 'notype']
                if not opp_types:
                    continue
                s = 0
                for otp in opp_types:
                    s += type_match(mt, otp)
                vals.append(s)
            return float(np.mean(vals)) if vals else 0.0

        p1_to_p2 = _avg_edge("p1_move_details", "p2_pokemon_state")
        p2_to_p1 = _avg_edge("p2_move_details", "p1_pokemon_state")
        f["type_edge_avg_diff"] = float(p1_to_p2 - p2_to_p1)
        f["p2_to_p1_type_edge_avg"] = float(p2_to_p1)
        p1_seen_types = set()
        for n in p1_seen:
            for tp in types.get(n, []):
                if tp != 'notype':
                    p1_seen_types.add(tp)
        p2_seen_types = set()
        for n in p2_seen:
            for tp in types.get(n, []):
                if tp != 'notype':
                    p2_seen_types.add(tp)
        f["p2_seen_type_count"] = int(len(p2_seen_types))
        f["type_seen_count_diff"] = int(len(p1_seen_types) - len(p2_seen_types))
        f["p2_damage_std"] = float(np.std(p2_loss)) if p2_loss else 0.0
        f["p2_damage_median"] = float(np.median(p2_loss)) if p2_loss else 0.0
        bp1 = [float(m.get("base_power")) for m in p1_moves if m.get("base_power") not in (None, "")]
        bp2 = [float(m.get("base_power")) for m in p2_moves if m.get("base_power") not in (None, "")]
        f["bp_std_p1"] = float(np.std(bp1)) if bp1 else 0.0
        f["bp_mean_p2"] = float(np.mean(bp2)) if bp2 else 0.0
        f["bp_std_diff"] = float(
            (np.std(bp2) if bp2 else 0.0) - (np.std(bp1) if bp1 else 0.0)
        )
        acc1 = [float(m.get("accuracy")) for m in p1_moves if m.get("accuracy") not in (None, "")]
        acc2 = [float(m.get("accuracy")) for m in p2_moves if m.get("accuracy") not in (None, "")]
        f["acc_mean_p2"] = float(np.mean(acc2)) if acc2 else 0.0
        f["acc_mean_diff"] = float(
            (np.mean(acc2) if acc2 else 0.0) - (np.mean(acc1) if acc1 else 0.0)
        )
        atk1 = sum(1 for m in p1_moves if (m.get("name") is not None))
        atk2 = sum(1 for m in p2_moves if (m.get("name") is not None))
        low1 = sum(1 for m in p1_moves if m.get("accuracy") not in (None, "") and float(m["accuracy"]) < 1.0)
        low2 = sum(1 for m in p2_moves if m.get("accuracy") not in (None, "") and float(m["accuracy"]) < 1.0)
        f["low_acc_share_diff"] = _safe_div(low2, max(1, atk2)) - _safe_div(low1, max(1, atk1))
        f["attacks_rate_diff"] = _safe_div(atk1, den) - _safe_div(atk2, den)

        def _edge_counts(side_key, opp_key):
            c_res, c_imm = 0, 0
            for t in tl:
                md = t.get(side_key) or {}
                if not md.get("name"):
                    continue
                mt = str(md.get("type", "")).lower()
                if not mt:
                    continue
                oppn = str(t[opp_key]["name"]).lower()
                opp_types = [tp for tp in types.get(oppn, ["notype", "notype"]) if tp != 'notype']
                for otp in opp_types:
                    v = type_match(mt, otp)
                    if v == -1:
                        c_res += 1
                    if v == -2:
                        c_imm += 1
            return c_res, c_imm

        p1_res, p1_imm = _edge_counts("p1_move_details", "p2_pokemon_state")
        p2_res, p2_imm = _edge_counts("p2_move_details", "p1_pokemon_state")
        f["resist_count_diff"] = int(p1_res - p2_res)
        f["p1_immune_count"] = int(p1_imm)
        f["immune_count_diff"] = int(p1_imm - p2_imm)

        def _boost_sum_pos(d):
            return sum(max(0, int(v)) for v in (d or {}).values())

        p1_boost_turns = sum(
            1 for t in tl if _boost_sum_pos(t["p1_pokemon_state"].get("boosts", {})) > 0
        )
        p2_boost_turns = sum(
            1 for t in tl if _boost_sum_pos(t["p2_pokemon_state"].get("boosts", {})) > 0
        )
        f["boost_turns_diff"] = int(p1_boost_turns - p2_boost_turns)
        f["p2_max_boost_sum"] = int(
            max((_boost_sum_pos(t["p2_pokemon_state"].get("boosts", {})) for t in tl), default=0)
        )
        last_p1 = str(tl[-1]["p1_pokemon_state"]["name"]).lower()
        last_p2 = str(tl[-1]["p2_pokemon_state"]["name"]).lower()
        t1 = [tp for tp in types.get(last_p1, ["notype", "notype"])]
        t2 = [tp for tp in types.get(last_p2, ["notype", "notype"])]
        tot = 0
        for a in t1:
            for b in t2:
                tot += type_match(a, b)
                tot -= type_match(b, a)
        f["types_last_round"] = int(tot)
        p1_kos = sum(1 for a, b in zip(p2_stat_raw, p2_stat_raw[1:]) if a != 'fnt' and b == 'fnt')
        p2_kos = sum(1 for a, b in zip(p1_stat_raw, p1_stat_raw[1:]) if a != 'fnt' and b == 'fnt')
        f["ko_rate_total"] = _safe_div(p1_kos + p2_kos, den)
        Esize = E
        p1_sw_late = (
            sum(1 for a, b in zip(p1_active[-Esize:], p1_active[-Esize + 1:]) if a != b)
            if den > 1 else 0
        )
        f["p1_switch_late_share"] = _safe_div(p1_sw_late, den)
        f["battle_id"] = battle.get("battle_id")
        if "player_won" in battle:
            f["player_won"] = int(battle["player_won"])
        out.append(f)
    return pd.DataFrame(out).fillna(0)


FEATURE_COLS_SUB1 = [
    "hp_edge_final","revealed_count_diff","status_severity_gap_final","ko_rate_total",
    "severe2_turns_diff","attacks_rate_diff","active_entropy_diff",
    "p2_used_count","par_turns_diff","type_edge_avg_diff","p2_late_damage","used_mean_spe_diff",
    "p1_status_mean_final","alive_diff_final","types_last_round","lead_type_edge","last_switch_turn_p1",
    "atk_edge_used","p1_turns_psn_tox","severe_status_early_share","p2_turns_psn_tox","initiative_early_diff",
    "bp_mean_p2","initiative_late_diff","first_blood_happened","run_len_mean_diff","heal_late_diff",
    "lead_def_edge","p2_max_boost_sum","p2_damage_std","p1_switch_late_share","p1_immune_count",
    "immune_count_diff","early_damage_gap","first_blood_side","severe_status_share","p2_turns_brn","hp_gap_mid",
    "hp_gap_var","bp_std_diff","hp_gap_early","p2_damage_median","p2_to_p1_type_edge_avg","p2_seen_type_count",
    "p1_turns_frz","low_acc_share_diff","acc_mean_diff","p2_run_len_mean","type_seen_count_diff",
    "mean_remaining_hp_p2","p1_mid_damage","p1_turns_par","boost_turns_diff","resist_count_diff",
    "p1_turns_brn","p1_mean_def","p2_switch_early_share","lead_type_fb_agree","hp_gap_sign_flips",
    "p1_used_mean_spe","p2_turns_slp","heal_mid_diff","p2_early_damage","bp_std_p1","acc_mean_p2",
    "lead_speed_fb_agree"
]



def create_features_sub1(train_data, test_data):
    train_df = create_features(train_data)
    test_df = create_features(test_data)
    X_train = train_df[FEATURE_COLS_SUB1].copy()
    y_train = train_df["player_won"].astype(int).copy()
    X_test = test_df[FEATURE_COLS_SUB1].copy()
    return X_train, y_train, X_test
