"""textarena — TextArena games as a v1 taskset, driven by a user simulator.

Each task is one episode of a TextArena game (the working example is Wordle). The model
plays by emitting moves; the framework's interception server drives a `vf.User` (the game
engine, `TextArenaUser` below) that replies with the game's feedback as a user turn — so a whole
game is one rollout of alternating assistant/user turns, and the harness never sees the
simulator. By default the user simulator runs in-process on the host (`UserConfig()`), spawning
no container or sandbox.

Scoring is game-authoritative: when the episode ends the user simulator records the game's
own outcome (`env.state.rewards`) onto `self.state.reward`, which `game_reward` reads off
`trace.state` — so the taskset needs no per-game guess parsing or container filesystem. Each
task is reproduced from an RNG seed (carried in `info`): the taskset seeds the game to build
the prompt and the simulator re-seeds to the same episode, so no per-game word-list or
state-key knowledge is needed and any single-player TextArena game fits.
"""

import copy
import random
from typing import Literal

import verifiers.v1 as vf

try:
    import nltk
    import textarena as ta
except ImportError as e:
    raise ImportError(
        "textarena requires nltk and textarena. Install with: uv add 'verifiers[ta]'"
    ) from e

SYSTEM_PROMPT = (
    "You are a competitive game player. Read the game instructions carefully and always "
    "use the exact move format the game requires. Think step-by-step first, then give your "
    "move as the only square-bracketed token in your reply — the game reads the first "
    "bracketed token, so don't put other words in brackets."
)


class TextArenaState(vf.State):
    game_over: bool = False
    reward: float = 0.0


class TextArenaUser(vf.User[vf.UserConfig, TextArenaState]):
    """The TextArena game engine as a framework-driven conversation partner. Holds one game in
    memory (set up from the task's `game` id + RNG `seed`, reproducing the taskset's episode) and,
    per `respond`, steps the game with the model's move and returns the next observation as a user
    turn. When the game ends it records the game's own outcome onto `self.state.reward`."""

    async def setup(self) -> None:
        import nltk

        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    async def setup_task(self, task) -> None:
        # textarena derives a game's whole setup from the global RNG at reset, so seeding it
        # reproduces the exact episode the taskset built the prompt from — no per-game keys.
        import random

        import textarena

        self.env = textarena.make(
            env_id=task.info["game"]
        )  # per-task input, from the task
        random.seed(task.info["seed"])  # per-task input
        self.env.reset(num_players=1)

    @staticmethod
    def _latest_feedback(observation: str) -> str:
        """Trim feedback to the latest block (after `Feedback:` in the last `[GAME]` message) so
        each injected user turn stays small and doesn't duplicate the history."""
        latest = observation.split("[GAME]")[-1].strip()
        return (
            latest.split("Feedback:")[-1].strip() if "Feedback:" in latest else latest
        )

    async def respond(self, message: str) -> vf.Messages:
        env = self.env
        env.step(
            message
        )  # TextArena parses the bracketed move out of the message itself
        if env.state.done:
            reward = float((env.state.rewards or {}).get(0, 0.0))
            reason = str(env.state.game_info[0]["reason"])
            self.state.reward = reward
            self.state.game_over = True
            return [{"role": "user", "content": reason}]
        _, observation = env.get_observation()
        return [{"role": "user", "content": self._latest_feedback(str(observation))}]


class TextArenaConfig(vf.TasksetConfig):
    game: Literal[
        "Wordle-v0",
        "Wordle-v0-long",
        "Hangman-v0",
        "WordLadder-v0",
        "WordSearch-v0",
    ]
    """The TextArena game (required). The tested single-player games: Wordle / Wordle-long
    and Hangman (guess a hidden word), WordLadder (change one letter at a time to reach a
    target), and WordSearch (find words in a grid)."""
    num_tasks: int = 1000
    """How many seeded episodes to generate; the eval/orchestrator selects from these."""
    user: vf.UserConfig = vf.UserConfig()
    """Where the TextArena user simulator runs (in-process on the host by default)."""


class TextArenaTask(vf.Task):
    info: dict
    """What the user simulator needs to set up the game: the `game` id and the RNG `seed`
    that reproduces the exact episode this task's prompt was built from."""


class TextArenaTaskset(vf.Taskset[TextArenaTask, TextArenaConfig, TextArenaState]):
    @vf.stop
    async def game_over(self, trace: vf.Trace) -> bool:
        return trace.state.game_over

    def load_tasks(self) -> list[TextArenaTask]:
        # One task per RNG seed; the simulator re-seeds to reproduce the same episode. Games
        # that embed the per-episode setup in the prompt (WordLadder's start/target,
        # WordSearch's grid) need the prompt built under each seed; games whose prompt
        # is seed-invariant (Wordle, Hangman) build it once.
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        template = ta.make(env_id=self.config.game)

        def observation(seed: int) -> str:
            random.seed(seed)
            env = copy.deepcopy(template)
            env.reset(num_players=1)
            return str(env.get_observation()[1])

        first = observation(0)
        seed_specific = observation(1) != first
        return [
            TextArenaTask(
                idx=i,
                name=f"{self.config.game}#{i}",
                prompt=observation(i) if seed_specific else first,
                system_prompt=SYSTEM_PROMPT,
                info={"game": self.config.game, "seed": i},
            )
            for i in range(self.config.num_tasks)
        ]

    def user(self, task: TextArenaTask) -> vf.User:
        return TextArenaUser(self.config.user)

    @vf.reward(weight=1.0)
    async def game_reward(self, trace: vf.Trace) -> float:
        return trace.state.reward


if __name__ == "__main__":
    TextArenaUser.run()
