# Speaker Notes — `.opencode` Harness Deck

Simple, spoken English for each slide. These are also embedded in the PowerPoint
itself: open the deck and choose **View → Notes** (or use Presenter View) to read
them while you present.

Tip: you don't need to read every word — these are just talking points. Speak
slowly and it's fine to pause.

---

## Slide 1 — Title
Hi everyone. Today I want to show you our pipeline — the multi-agent system we
built under the `.opencode` folder. In short, it lets an AI optimize our kernel
code, but in a safe, step-by-step way. It works like a small team: one agent
researches, one plans, one writes code, one reviews, and one tests on a real
phone. Let's get started.

## Slide 2 — Agenda
Here is what I'll cover. First, the big picture — where this system sits. Then
why we built it this way, and the three ways to start a job. After that, the
main pipeline and its safety gates. Then how the agents talk to each other, how
it tests on real hardware, how it remembers things, and finally how you actually
run it. I'll keep it simple.

## Slide 3 — The big picture
There are two parts. `hmopt` is the platform — the tools, the code index, the
database, the APIs. `.opencode` is the brain — the agents that decide what to do.
The agents reach the tools through six MCP servers, shown in the middle. At the
bottom is the real target: our kernel source and a real phone. Today we mostly
talk about the brain — the `.opencode` part.

## Slide 4 — Why a staged harness
Why not just one big AI agent? Because one agent fails in four common ways. It
drifts — it starts coding when it should still be researching. It skips testing
— code lands with no proof. It forgets — long chats lose memory. And its "wins"
can't be trusted, because there are no real numbers. Our system fixes each one.
We measure "instruction count" because it is stable and easy to check — same
input, same number.

## Slide 5 — Three entry routes
There are three ways to start a job. One: the full auto pipeline — you give a
target and it does everything end to end. Two: human-in-the-loop — you and the
AI research and plan together, step by step. Three: a deep dive on one single
function, just to understand it. Most of the time we use route one. Routes two
and three feed into route one.

## Slide 6 — The 7-stage gated pipeline
This is the heart of the system. The green path is the normal flow: intake,
research, plan review, write code, code review, test, decide. The orange boxes
are gates — you cannot skip them. The red arrows show what happens when something
fails. If a review or a test fails, the work goes back to the right owner — not
random, always one owner. There is also a limit on retries, so it can't loop
forever. If it hits the limit, it stops and asks a human.

## Slide 7 — Hub-and-spoke
One agent is the boss — the manager in the middle. Only the manager hands out
work. Every other agent does its job, then comes back to the manager. The manager
reads the result and sends out the next task. One thing we guard against: the
manager faking it — just writing "I delegated to X" instead of really doing it.
So we added simple checks to make sure a real agent actually ran.

## Slide 8 — Two gates + the A/B rule
Two gates you can't skip. Plan review: before any code, someone checks the plan.
Code review: after the code, someone checks it. And the test rule: we must flash
and test both versions — the old one and the new one. One test alone does not
count. Then we compare. If the new one uses fewer or equal instructions, it
passes. If it uses more, it fails.

## Slide 9 — Closed-loop self-correction
Same idea as before, with a bit more detail. When a stage fails, it goes back to
one owner upstream, with a limit on tries. On the right is Auto-Iterate. If a run
passes cleanly, it can start again on the same target to find more wins. It
treats the past wins as already done, and looks for new, different ideas. It
stops when it runs out of ideas or stops improving.

## Slide 10 — The handoff packet
When one agent finishes, it passes a small "packet" to the next one. Every
handoff carries the same common fields — shown on the left: the target, the
files, the risks, the next action. On top of that, each step adds a few of its
own — that's the table on the right. For example, the coder adds the list of
changed functions; the reviewer adds the two image paths and the compare level;
the tester adds the verdict and the delta. The big stuff stays on disk; the
packet just points to it. So the chat stays short, and a long run stays easy to
follow.

## Slide 11 — Skill packs
Skills are just shared rule files. When you run a command, these skills are
loaded into the manager once. Then every agent can see them. So the agents don't
re-read them — they already have them. We group them by job: the goal, the
process, the gates, testing, and memory. About thirteen skills load on a full run.

## Slide 12 — Research & ideation
Before any idea, the agent follows a fixed order: think first, use the code index,
read the source, build a clear picture, and only then propose a fix. For ideas,
it makes exactly five, drops the ones we already tried, ranks them, and picks the
best. The table shows our research agents — who talks to a human, who writes a
plan, and so on. Don't worry about every row; the point is each one has a clear
job.

## Slide 13 — A/B validation on real hardware
This is the cool part — it tests on a real phone. First, build and sign the new
image. Flash the old version, wait about ten minutes to settle, then test it.
Then flash the new version, wait again, test, and compare. We always wait after
each flash, and we never run both at the same time. If the build fails, that's a
fail. If the phone or the cable fails, that's "skipped" — not the code's fault.

## Slide 14 — Cross-run memory + idea ledger
The system remembers things across runs, so it gets smarter over time. There is
memory by area — file, subsystem, and global lessons. There is the idea ledger —
every idea we tried, with its status: approved, landed, or rejected. And there is
a log of human decisions. Nothing gets deleted. So next time it won't suggest
something we already rejected.

## Slide 15 — Compaction-proof by design
Long AI runs lose memory when the chat gets cut. We planned for that. The rule is
simple: the file wins over the chat. The current state lives in one file,
`current_task.json`. The agent writes to disk before it acts. So if a session
dies, a new one just reads the files and keeps going. On the right is a real
example of that state file.

## Slide 16 — The MCP toolbelt
These are the six tool servers the agents use. Sequential Thinking — to plan.
Kernel Index — to search the code. Build — to compile and sign. Flash — to put
the image on the phone. Auto-Test — to run the tests. Git — to see changes. Each
one has its own port. One nice detail: the Kernel Index is built by the hmopt
platform — that's where the two halves meet.

## Slide 17 — How to run it
Here is how you actually use it. Open the tool from the kernel folder. Edit a
command — set the target, the goal, and how many rounds. Type the slash command,
like `/optimize_generic`. Then the manager runs the whole pipeline for you. The
table shows ready-made presets for different parts of the kernel. There are also
commands just for research or planning. And you can switch the language in one
config file.

## Slide 18 — Artifact trail
Every run leaves files behind. Design notes, the plan, the two review results,
the patch, the test results, the memory, and the state file. So a run is not a
black box — you can open these files and see exactly what happened, and even
replay it. It's all there on disk.

## Slide 19 — Takeaways
To wrap up. A staged, gated pipeline makes the AI work like a careful team. Two
hard gates plus a real hardware test — no code without a plan, no win without
proof. One manager runs the loop and fixes its own mistakes. It is built for long
runs, and it learns over time. And you start it all with one command. That's it —
happy to take any questions.
