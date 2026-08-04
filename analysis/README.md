# analysis/

Research campaigns that outgrew this repository now live in their own public repositories, so the
platform repo carries the site and not the science:

| Campaign | Repository |
|---|---|
| arXiv topics — the model of record behind `/topics` | https://github.com/ArtaQuest/artatopics |
| ArtaMusic — the generation pipeline | https://github.com/ArtaQuest/artamusic |
| Arta — the mascot's rig and scenes | https://github.com/ArtaQuest/artalife |

Each was moved after verifying, file by file, that the destination repository held every tracked
file. Copies lingered here afterwards and drifted: a reader landing on the version in this
repository was reading a snapshot that had stopped being updated, with nothing saying so.

`analysis/citations` stays — it is a shared platform dataset the site itself reads, not a campaign.
