"""Prompt construction for Cocktail-DialogGen's Gemini dialog composer.

Generalized to N speakers (the agent + N-1 other speakers), with explicit control
over the agent's gender and, optionally, the other speakers' genders.
"""
from __future__ import annotations

from typing import List, Optional

VIBES = {
    "assisting": (
        "ASSISTING: The Agent is a formal or semi-formal assistant to help the other speakers. "
        "The tone is professional, efficient, and helpful. Crucially, the Agent does not know and "
        "will not make up quantitative numbers or tech specs, including exact temperatures, "
        "coordinates, percentages, and etc."
    ),
    "casual": (
        "CASUAL: The Agent is an acquaintance of the other speakers. The tone is relaxed, natural, "
        "and conversational."
    ),
}

NAMES_POLICIES = {
    "named": (
        "NAMED: Assign a unique name to every speaker. Use these names consistently (but not "
        "necessarily in every turn) throughout the dialogue to anchor identities and clarify who "
        "is being addressed."
    ),
    "anonymous": (
        "ANONYMOUS: Strictly avoid mentioning any names for any speaker. Speakers identify their "
        "target through second-person address or by maintaining the logical thread of the current "
        "exchange."
    ),
}

# speaker_b, speaker_c, speaker_d, ... (the agent is always "agent")
_OTHER_KEYS = ["speaker_b", "speaker_c", "speaker_d", "speaker_e", "speaker_f"]


def other_speaker_keys(n_speakers: int) -> List[str]:
    """Return the non-agent speaker keys for an n_speakers conversation."""
    if n_speakers < 2:
        raise ValueError("n_speakers must be >= 2 (the agent plus at least one other speaker).")
    return _OTHER_KEYS[: n_speakers - 1]


def caption_prompt() -> str:
    """Instruction for Gemini to caption an environment recording in one sentence."""
    return (
        "Listen to this audio recording of an acoustic environment. In ONE concise sentence, "
        "describe the place/setting and the characteristic background sounds (e.g., "
        "'A busy hospital cafeteria with clattering trays, overlapping chatter, and distant "
        "announcements'). Output only the sentence, with no preamble or quotation marks."
    )


def _gender_clause(agent_gender: str, other_genders: Optional[List[str]], other_keys: List[str]) -> str:
    lines = [f"- Agent: the human listener/speaker you are modeling. The Agent's gender MUST be {agent_gender}."]
    if other_genders is not None:
        if len(other_genders) != len(other_keys):
            raise ValueError(
                f"other_genders has {len(other_genders)} entries but there are {len(other_keys)} "
                f"other speakers."
            )
        for key, g in zip(other_keys, other_genders):
            lines.append(f"- {key}: a human speaker whose gender MUST be {g}.")
    else:
        for key in other_keys:
            lines.append(f"- {key}: a human speaker who may be female or male (assign without bias).")
    return "\n".join(lines)


def build_composer_prompts(
    *,
    n_speakers: int,
    environment_description: str,
    vibe: str,
    names_policy: str,
    agent_gender: str = "female",
    other_genders: Optional[List[str]] = None,
    n_dialogs: int = 10,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the Gemini dialog composer."""
    vibe = vibe.lower()
    names_policy = names_policy.lower()
    if vibe not in VIBES:
        raise ValueError(f"vibe must be one of {list(VIBES)}, got {vibe!r}")
    if names_policy not in NAMES_POLICIES:
        raise ValueError(f"names_policy must be one of {list(NAMES_POLICIES)}, got {names_policy!r}")

    other_keys = other_speaker_keys(n_speakers)
    all_keys = ["agent"] + other_keys
    gender_clause = _gender_clause(agent_gender, other_genders, other_keys)

    # voice_profiles skeleton for the output format
    profiles_lines = []
    for key in all_keys:
        gender_hint = agent_gender if key == "agent" else "<male_or_female>"
        profiles_lines.append(
            f'        "{key}": {{\n'
            f'            "gender": "{gender_hint}",\n'
            f'            "name": "<random_real_name>",\n'
            f'            "role": "<social_role>",\n'
            f'            "style": "<brief personality + speaking style>"\n'
            f'        }}'
        )
    profiles_block = ",\n".join(profiles_lines)

    system_prompt = (
        "You are a synthetic dialog generator. Create realistic multi-talker conversation logs "
        f"within the following environment: {environment_description}"
    )

    speaker_list = ", ".join(["Agent"] + [k.replace("speaker_", "Speaker ").upper() for k in other_keys])
    user_prompt = f"""***Setting***

This conversation has {n_speakers} speakers who can all hear each other:
{gender_clause}

Conversation Vibe: {VIBES[vibe]}

Speakers' Names: {NAMES_POLICIES[names_policy]}

Environment: {environment_description}

***Conversation Logic***

1. Multi-Talker Conversation: All {n_speakers} speakers ({speaker_list}) talk to each other and can hear each other.

2. Speakers' Roles: Each speaker has a real (not virtual) well-defined social role relevant to the setting (e.g., job title, relationship, responsibility). Their roles must be reflected in what they talk about and how they address each other. Avoid vague or generic labels (e.g., "ticket holder", "passenger", "customer", "employee").

3. Speakers' Styles: Each speaker also has a style attribute that includes both personality (e.g., curious, considerate, extroverted/introverted) and speaking style (e.g., concise, playful, slow/fast). Personality defines their general disposition, while speaking style defines how they express themselves.

4. Dialog Directions: All directions are possible: between the Agent and any other speaker, and between two other speakers (overheard by the Agent).

5. Dialog Switch: Every conversation must include at least one addressee switch, where the Agent moves from replying to one speaker to replying to a different speaker. Both are <RESPOND> turns; the switch is expressed through the conversation content and addressing, not a special action tag.

6. Natural Addressee Clarity Rule: Every turn must make the intended addressee clear and unambiguous, but speakers should do this naturally through pronouns, callbacks, turn-taking, topic continuity, or shared context. Do not repeatedly address people by roles, titles, or names. They may be used only on first mention (or not used at all), or when it is genuinely necessary to avoid ambiguity.

***Agent's Actions***

For each turn, the Agent takes exactly one of the following three actions:

<RESPOND>: The current speaker directly addresses the Agent. The Agent replies to that speaker. This also covers the case where a speaker redirects the conversation back to the Agent (an addressee switch): the Agent still replies, and the action is <RESPOND>.

<LISTEN>: Two other speakers talk exclusively to each other, not to the Agent and not expecting the Agent to respond. This usually happens when one speaker brings up a topic outside the Agent's role or relevance, and/or uses clear linguistic cues to address the other speaker instead of the Agent. The Agent stays silent.

<IGNORE>: Background noise only. No one speaks to the Agent. The Agent stays silent.

Each dialog must include all three actions: <RESPOND>, <LISTEN>, and <IGNORE>.

***Generation Constraints***

1. Speaker field: Must be exactly one of {other_keys} or "noise".

2. Literal Speech: No narration or internal thought.

3. Noise: The spoken conversation happens under the background noise of the environment described above.

4. Agent Silence: For <LISTEN> and <IGNORE>, the Agent does not speak, so agent_text must be null (do NOT output any placeholder text).

5. Timing Logic: Assign pause_before_input for each dialog turn. Reference: 0.0-0.5s for fluid conversation and reactive replies; 0.5-2.0s for shallow thought; 2.0-5.0s for deep thought, switching the topic, environmental distractions, or physical tasks.

***Output Format***

Output a JSON file (no markdown, no commentary). The output must be a list of conversations. Each conversation must be a dictionary with exactly two keys:

'setting' contains the metadata for the environment and speakers:
{{
    "environment": "{environment_description}",
    "activity": "<specific activity>",
    "voice_profiles": {{
{profiles_block}
    }}
}}

'dialog' contains a list of interaction turns. Each speech turn looks like:
{{
    "action": "<RESPOND> or <LISTEN>",
    "input_speaker": "one of {other_keys}",
    "pause_before_input": <seconds the input speaker waits after the previous turn, 0~5>,
    "input_text": "<literal speech>",
    "agent_text": "<agent reply for <RESPOND>; null for <LISTEN>>"
}}
and occasionally when no one is talking (a temporary standstill or the speakers are busy):
{{
    "action": "<IGNORE>",
    "input_speaker": "noise",
    "pause_before_input": <seconds the noise alone occupies the audio, 2~10>,
    "input_text": null,
    "agent_text": null
}}

Each conversation can start either with a speech turn or a noise turn.

***Task***

Generate {n_dialogs} unique conversations that often happen in this environment, as diverse as possible, with 8-20 turns per conversation.
Ensure all speakers interact significantly with each other, and all three Agent's actions (<RESPOND>, <LISTEN>, <IGNORE>) are well represented, including at least one addressee switch.
The Agent's gender must be {agent_gender}, and every speaker's gender must be assigned as specified above, without bias or stereotypical assumptions.
Dialogs must not include harmful, racist, discriminatory, hateful, or otherwise abusive content.
Prevent any specific numbers, coordinates, or tech specs in the dialogs, as the Agent does not know and will not make up such quantitative information.
Make sure you follow all the generation constraints and the output JSON structure.
"""
    return system_prompt, user_prompt
