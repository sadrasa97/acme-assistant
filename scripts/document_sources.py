"""Reproducible source text for the sample Acme Analytics PDFs.

Each document is a list of sections; every section is rendered on its own page so that
page numbers in citations are meaningful. Edit this file and re-run
`python -m scripts.generate_documents` to regenerate the PDFs.
"""

DOCUMENTS = {
    "refund_policy.pdf": {
        "title": "Acme Analytics Refund and Cancellation Policy",
        "version": "Version 3.2 - effective 1 January 2026",
        "sections": [
            (
                "1. Refund eligibility",
                [
                    "Acme Analytics offers a 30-day money-back guarantee on all new annual subscriptions. "
                    "A customer may request a full refund within 30 calendar days of the original purchase date, "
                    "provided the account has used fewer than 5,000 dashboard queries during that period.",
                    "Monthly subscriptions are refundable only within the first 7 days of the initial purchase. "
                    "Renewal payments for monthly plans are not refundable, but the customer may cancel at any time "
                    "to avoid the next billing cycle.",
                    "Professional services such as onboarding workshops and custom connector development are "
                    "non-refundable once work has started. Unused prepaid service hours are refunded in full if "
                    "the customer cancels at least 10 business days before the scheduled start date.",
                ],
            ),
            (
                "2. How to request a refund",
                [
                    "Refund requests must be submitted through the Billing page of the customer portal or by "
                    "emailing billing@acme-analytics.example with the order identifier. Requests sent to sales "
                    "representatives or through social media are not processed.",
                    "The Billing team acknowledges every request within 2 business days. Approved refunds are "
                    "issued to the original payment method within 10 business days of approval. Refunds for "
                    "payments made by bank transfer are issued by transfer and may take up to 15 business days.",
                    "Refunds above 5,000 USD require a second approval from the Finance Manager before they are "
                    "released.",
                ],
            ),
            (
                "3. Cancellation and partial periods",
                [
                    "Annual subscriptions cancelled after the 30-day guarantee window are not refunded, but the "
                    "service stays active until the end of the paid term. Acme Analytics does not prorate annual "
                    "plans except where required by local law.",
                    "If Acme Analytics suspends an account because of a verified service outage longer than 24 "
                    "consecutive hours, the customer receives a service credit equal to one day of the "
                    "subscription fee for each full day of outage.",
                ],
            ),
        ],
    },
    "discount_policy.pdf": {
        "title": "Acme Analytics Sales and Discount Policy",
        "version": "Version 2.4 - effective 1 March 2026",
        "sections": [
            (
                "1. Standard discount tiers",
                [
                    "Sales representatives may offer volume discounts based on the annual contract value (ACV). "
                    "Contracts with an ACV from 10,000 to 24,999 USD receive a 5 percent discount. Contracts from "
                    "25,000 to 99,999 USD receive a 10 percent discount. Contracts of 100,000 USD or more receive "
                    "a 15 percent discount.",
                    "Multi-year prepayment earns an additional 3 percent discount for a two-year commitment and "
                    "an additional 5 percent for a three-year commitment. Discounts from different tiers are "
                    "additive, but the total discount may never exceed 25 percent without written approval.",
                ],
            ),
            (
                "2. Special programs",
                [
                    "Verified non-profit organizations and accredited educational institutions receive a flat "
                    "20 percent discount on all plans. This discount cannot be combined with volume discounts. "
                    "Eligibility must be confirmed with a registration document before the order is created.",
                    "Startups with fewer than 25 employees and less than 2 million USD in funding may apply for "
                    "the Launch Program, which provides 50 percent off the first 12 months of the Starter plan.",
                    "Customers who refer a new paying account earn a one-time referral credit of 500 USD once "
                    "the referred customer has completed 60 days on a paid plan.",
                ],
            ),
            (
                "3. Approval rules",
                [
                    "Any discount above 15 percent requires approval from the Regional Sales Director. Any "
                    "discount above 25 percent requires approval from the VP of Sales. Approvals must be "
                    "recorded in the CRM before a quote is sent to the customer.",
                    "Promotional coupon codes are valid for one order only and expire 90 days after issue. "
                    "Coupons cannot be applied retroactively to orders that have already been invoiced.",
                ],
            ),
        ],
    },
    "customer_support_handbook.pdf": {
        "title": "Acme Analytics Customer Support Handbook",
        "version": "Internal - revision 5 - 15 February 2026",
        "sections": [
            (
                "1. Support channels and hours",
                [
                    "Customer support is available by email, live chat and the customer portal. Live chat and "
                    "phone support operate Monday to Friday from 08:00 to 18:00 UTC. Email and portal tickets "
                    "are monitored around the clock for Enterprise customers and during business hours for all "
                    "other plans.",
                    "Premium Support customers also have access to a dedicated Slack channel with a named "
                    "customer success manager.",
                ],
            ),
            (
                "2. Response time targets",
                [
                    "Ticket priority determines the first-response target. Priority 1 (service down) must be "
                    "answered within 1 hour, 24 hours a day. Priority 2 (major feature impaired) must be answered "
                    "within 4 business hours. Priority 3 (minor issue or question) must be answered within 1 "
                    "business day. Priority 4 (feature request) must be answered within 3 business days.",
                    "Agents must update the customer at least every 2 hours on open Priority 1 tickets until "
                    "service is restored.",
                ],
            ),
            (
                "3. Escalation procedure",
                [
                    "A ticket is escalated to a Tier 2 engineer if it is not resolved within 8 business hours or "
                    "if it involves data accuracy, security or billing disputes. Tier 2 escalates to the on-call "
                    "engineering manager when a Priority 1 ticket remains open after 4 hours.",
                    "Security incidents must be reported to the security team immediately through the internal "
                    "incident form, and customer-facing agents must not discuss technical details of an incident "
                    "until the security team approves the communication.",
                ],
            ),
        ],
    },
}
