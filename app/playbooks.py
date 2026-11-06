"""Reviewer notes by denial type: what a strong appeal usually has to address, and what to ask for.

These are working suggestions for the person reviewing the draft (an advocate, biller or the patient). They are NOT
facts about the patient and are never inserted into the letter as claims. They contain no legal citations; rights
statements still come only from the cited government passages.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Playbook:
    title: str
    address: tuple[str, ...]        # points the appeal should respond to
    ask_clinician: tuple[str, ...]  # what to request from the treating clinician
    ask_plan: tuple[str, ...]       # what to request from the insurer


PLAYBOOKS: dict[str, Playbook] = {
    "medical_necessity": Playbook(
        "Denied as not medically necessary",
        address=(
            "Answer the plan's stated reason point by point, using the clinical criteria it relied on.",
            "Show what was already tried (therapy, medication, imaging) and why the requested service is the next step.",
            "Tie the request to this patient's symptoms, findings and dates, not to general statements.",
        ),
        ask_clinician=(
            "A letter of medical necessity that addresses each element of the plan's criteria.",
            "Chart notes and test results that support it.",
            "Willingness to take part in a doctor-to-doctor (peer-to-peer) review.",
        ),
        ask_plan=(
            "The exact clinical criteria or guideline used, and the reviewer's specialty.",
            "The complete claim file.",
            "Whether a peer-to-peer review is available before the appeal deadline.",
        ),
    ),
    "prior_authorization": Playbook(
        "Denied for missing or invalid prior authorization",
        address=(
            "Establish whether authorization was requested, when, and by whom.",
            "If it was not, explain why (emergency, provider error, plan error) and ask for retroactive review.",
            "Show the service itself meets coverage criteria, in case the plan reviews it on the merits.",
        ),
        ask_clinician=(
            "Confirmation of whether and when an authorization request was submitted, with any reference number.",
            "A letter supporting a retroactive authorization if one is possible.",
        ),
        ask_plan=(
            "Its record of any authorization request, with dates.",
            "Whether retroactive authorization is allowed and how to request it.",
        ),
    ),
    "out_of_network": Playbook(
        "Denied as out of network",
        address=(
            "Say whether the care was an emergency, or whether no suitable in-network provider was reasonably available.",
            "Ask for a network exception or in-network cost sharing where that applies.",
            "Check that the plan's own provider directory and the provider's network status agree.",
        ),
        ask_clinician=(
            "A letter explaining why this provider was needed and why in-network options did not fit.",
        ),
        ask_plan=(
            "The network exception process and any form it requires.",
            "Proof of how it determined the provider's network status on the date of service.",
        ),
    ),
    "experimental": Playbook(
        "Denied as experimental or investigational",
        address=(
            "Respond to the plan's definition of experimental and the evidence it says is missing.",
            "Show published evidence and professional guidance supporting the service for this condition.",
            "Explain why standard alternatives were unsuitable or already tried.",
        ),
        ask_clinician=(
            "A letter with the supporting studies or guidelines, and why alternatives are not appropriate.",
        ),
        ask_plan=(
            "Its written definition of experimental and the specific evidence review it relied on.",
            "The credentials of the reviewer who made the decision.",
        ),
    ),
    "limit_reached": Playbook(
        "Denied because a benefit limit was reached",
        address=(
            "Check that the plan counted visits or units correctly, in the right plan year.",
            "If the count is right, request a medical exception to the limit with clinical support.",
        ),
        ask_clinician=(
            "A letter explaining why more services are needed than the limit allows.",
            "Progress notes that show benefit from the care so far.",
        ),
        ask_plan=(
            "An itemized count of the visits or units applied against the limit.",
            "The process for an exception to the limit.",
        ),
    ),
    "not_covered": Playbook(
        "Denied as not a covered benefit",
        address=(
            "Find the exact exclusion language in the plan document and compare it to the service billed.",
            "Check whether a different code, setting or covered alternative fits what was actually done.",
        ),
        ask_clinician=(
            "Confirmation that the billing code matches the service, or a corrected claim if it does not.",
        ),
        ask_plan=(
            "The plan-document language relied on, with the page or section.",
            "Whether any exception applies.",
        ),
    ),
    "general": Playbook(
        "Denial, reason not clearly categorised",
        address=(
            "Restate the plan's own reason in your words and respond to it directly.",
            "Keep to dates, codes and facts found in the denial letter.",
        ),
        ask_clinician=("Supporting notes or a letter relevant to the stated reason.",),
        ask_plan=(
            "The complete claim file and the criteria used.",
            "A written explanation of the decision.",
        ),
    ),
}


def playbook_for(kind: str) -> Playbook:
    return PLAYBOOKS.get(kind, PLAYBOOKS["general"])
