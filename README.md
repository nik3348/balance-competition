# Game Balance Competition

Finding parameter settings that make four board games as balanced as possible,
scored by a local server at `http://localhost:3000`.

## How the search works

Everything is built on one API call. `POST /api/run_game` takes a game, a params
dict and a run type, and returns a single balance score.

The space is far too big to search exhaustively. CantStop has 13 parameters with
5 legal values each, so 1.2 billion combinations; Wonders7 has 7.2 x 10^20. At
2.5 minutes per evaluation, a full sweep of CantStop would take 5,000 years.

So the search uses **coordinate descent**: tune one parameter at a time while
holding all the others fixed. Try every legal value for parameter 1, keep the
best, move to parameter 2, and so on. Once you have been through them all, go
round again. Stop when a full pass changes nothing.

That turns a multiplication into an addition. Instead of 1.2 billion
combinations, one pass over CantStop costs 52 evaluations.

Each run has four phases:

1. **Seed** — start from the best params found so far.
2. **Random search** — try 3-5 random settings, in case the seed is in a bad region.
3. **Coordinate descent** — the sweeps described above. All candidate values for
   a given parameter are tested in parallel across 3-4 threads, with retries,
   because the server drops requests during long runs.
4. **Verification** — re-run the winner 20 times and take the mean. The scorer is
   noisy enough that a single evaluation cannot be trusted on its own.

## The files

**Core**

| File | What it does |
|---|---|
| `api_client.py` | The only code that talks to the server. Everything else imports it. |
| `valid_params.json` | The legal values for every parameter of every game. Defines the search space. |

**Optimizers**

| File | What it does |
|---|---|
| `opt.py` | The main search. All four phases. Produced the Dominion, Wonders7 and ExplodingKittens params. |
| `final_opt.py` | A second pass that skips random search and starts from known-good params. Produced the CantStop result. |
| `verify.py` | No searching. Runs two param sets 20 times each and compares their averages. The tiebreaker. |
| `smart_optimizer.py` | Earlier search that found CantStop's key setting: `COLUMNS_TO_WIN=2` with `MARKERS=2`, which scores highest and also evaluates fastest (78s vs 200s). Every later run starts from it. |

## Running it

The server must be up at `localhost:3000` first.

```sh
# full search for one game
uv run python opt.py --game Wonders7

# second pass from known-good params
uv run python final_opt.py --game CantStop

# compare two param sets over 20 runs each
uv run python verify.py --game Dominion --n-reps 20
```

Useful flags: `--workers` (parallel evaluations, default 4), `--n-verify`
(verification reps, default 10 in `opt.py`, 20 in `final_opt.py`),
`--n-random` (random search samples), `--max-rounds` (cap on sweeps).

A full search takes 6-14 hours per game. Run it under `nohup` or `tmux`.
