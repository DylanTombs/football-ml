# Press-conference text (plan 08)

Paste each manager's pre-match press conference or team news into
`gwNN/<team-slug>.md` (or `.txt`) before running `predict`. One file per team.

- Slugs are the canonical team name in lower case with spaces as dashes and no
  apostrophes: `arsenal`, `man-united`, `nottm-forest`, `west-ham`. A wrong name is
  skipped with a warning that lists the valid ones.
- Claude reads only what's pasted and reports availability (out / doubtful / fit) for
  players it mentions, by FPL id. It feeds the shadow prediction only.
- Commit the files with the predictions, so every shadow run can be repeated.
