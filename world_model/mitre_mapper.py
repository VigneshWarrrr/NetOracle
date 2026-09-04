"""
MITRE ATT&CK Stage Mapping and Narrative Generation
"""
from typing import Dict, List

class MITREMapper:
    STAGES = {
        0: {
            'name': 'Benign',
            'tactic': 'Normal Operations',
            'description': 'Normal network traffic with no detected malicious activity.',
            'color': '#2ecc71',
            'severity': 0,
            'mitre_id': 'N/A',
            'recommended_action': 'Continue standard network monitoring.'
        },
        1: {
            'name': 'Reconnaissance',
            'tactic': 'TA0043',
            'description': 'Attacker gathering network topology and service profiles.',
            'color': '#f39c12',
            'severity': 2,
            'mitre_id': 'TA0043',
            'recommended_action': 'Inspect source IP scan velocity. Enforce border rate-limiting.'
        },
        2: {
            'name': 'Initial Access',
            'tactic': 'TA0001',
            'description': 'Attempts to gain initial foothold via brute force or exposed services.',
            'color': '#e67e22',
            'severity': 4,
            'mitre_id': 'TA0001',
            'recommended_action': 'Isolate targeted perimeter hosts. Throttle authentication endpoints.'
        },
        3: {
            'name': 'Execution',
            'tactic': 'TA0002',
            'description': 'Execution of adversary-controlled code and payloads.',
            'color': '#e74c3c',
            'severity': 6,
            'mitre_id': 'TA0002',
            'recommended_action': 'Quarantine affected host process space. Trigger memory snapshot.'
        },
        4: {
            'name': 'Lateral Movement',
            'tactic': 'TA0008',
            'description': 'Adversary extending access across internal subnets and credentials.',
            'color': '#c0392b',
            'severity': 8,
            'mitre_id': 'TA0008',
            'recommended_action': 'Segment VLANs immediately. Revoke high-privilege session tokens.'
        },
        5: {
            'name': 'Command & Control',
            'tactic': 'TA0011',
            'description': 'Maintaining persistent communication channel to external infrastructure.',
            'color': '#8e44ad',
            'severity': 9,
            'mitre_id': 'TA0011',
            'recommended_action': 'Sinkhole identified C2 domains. Block outbound beaconing egress.'
        },
        6: {
            'name': 'Exfiltration',
            'tactic': 'TA0010',
            'description': 'Unauthorized data staging and exfiltration out of network perimeter.',
            'color': '#2c3e50',
            'severity': 10,
            'mitre_id': 'TA0010',
            'recommended_action': 'CRITICAL ALERT: Sever non-essential external bandwidth. Engage Incident Response.'
        }
    }

    @classmethod
    def get_stage_info(cls, stage_id: int) -> Dict:
        return cls.STAGES.get(int(stage_id), cls.STAGES[0])

    @classmethod
    def get_stage_name(cls, stage_id: int) -> str:
        return cls.STAGES.get(int(stage_id), cls.STAGES[0])['name']

    @classmethod
    def generate_kill_chain_narrative(cls, stage_sequence: List[int]) -> str:
        if all(s == 0 for s in stage_sequence):
            return "All forecast windows project benign baseline network traffic."
        
        narratives = []
        for step, stage in enumerate(stage_sequence):
            if stage > 0:
                info = cls.get_stage_info(stage)
                narratives.append(f"- **T+{step+1}**: {info['name']} ({info['mitre_id']}) — *{info['description']}*")
        return "\n".join(narratives)
