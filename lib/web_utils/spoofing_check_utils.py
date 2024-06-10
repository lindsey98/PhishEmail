import re
from ipaddress import ip_address
from typing import Tuple
import dkim
from email.message import EmailMessage
import os
import subprocess
import dns.resolver
import ipaddress

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
def extract_sender_ip(email_headers: Tuple) -> str:
    # Regular expression pattern to extract IP addresses from "Received" lines
    ip_regex = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    ips = []
    for header in email_headers:
        if header[0].startswith('Received'):
            found_ips = ip_regex.findall(header[1])
            ips.extend(found_ips)

    # Iterate over the found IPs in reverse order (start from the original sender)
    for ip in reversed(ips):
        try:
            # Check if the IP address is a private address
            if not ip_address(ip).is_private:
                return ip
        except ValueError:
            # If IP address is invalid, continue to the next
            continue

    return "NotFound"


def parse_spf_record(spf_record, ip):
    # This function needs to handle includes, redirects, IP ranges, etc.
    parts = spf_record.split()
    for part in parts:
        if part.startswith("ip4:") or part.startswith("ip6:"):
            network = part.split(':')[1]
            if ipaddress.ip_address(ip) in ipaddress.ip_network(network, strict=False):
                return True
        elif part == "all":
            return False  # typically means any IP not previously matched should fail
    return False  # Default deny if not matched or improperly formatted

def check_spf(sender_claimed_domain, sender_ip):
    try:
        answers = dns.resolver.resolve(sender_claimed_domain, 'TXT')
        for record in answers:
            text = record.to_text()
            if text.startswith('"v=spf1') or text.startswith("v=spf1"):
                spf_record = text.strip('"')
                return parse_spf_record(spf_record, sender_ip)
        # No SPF record found
        print(f"No valid SPF record found for domain: {sender_claimed_domain}")
        return True
    except dns.resolver.NoAnswer:
        print("No SPF record available.")
        return True  # Consider how to handle this based on your policy
    except dns.resolver.LifetimeTimeout:
        print("Query timed out.")
        return True
    except dns.resolver.NoNameservers:
        print("No nameservers available.")
        return True
    except Exception as e:
        print(f"DNS query failed: {e}")
        return True