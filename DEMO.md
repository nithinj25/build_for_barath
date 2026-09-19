# Demo script — 3 minutes

Live: https://56z73fanvpxorv5dmkbc5d2kru0vmkqz.lambda-url.ap-south-1.on.aws/
(synthetic data; demo mode shows green "true link (synthetic truth)" marks so
the audience can see when the system is right and when it is not).

Every number you say is in [FINDINGS.md](FINDINGS.md). Say "on synthetic
data" once, early, and mean it.

## 0:00 — The problem (20 s)

> CCTNS and ICJS already move FIRs between states. What nobody computes is
> "this burglary in Mysuru looks like that one three weeks later, filed at a
> different police station." An officer can search once they suspect a link.
> This finds the links nobody suspected yet.

## 0:20 — Leads inbox (40 s)

Open the page. It lands on **Leads → Same district**.

- Point at the blue line: *about 1 in 4 leads like these was a real serial
  link in testing — ~6,200× better than picking case pairs at random.*
- Lead #1: house burglary, Mysuru, **different police stations**, 3 weeks
  apart. Both FIRs: gas cutter (4% of house burglaries), faces covered,
  inside information.
- Click **Other states**: the yellow warning. *We don't hide cross-state
  matches — no keyword search can reach them — but in testing they were
  almost never real, and the screen says so.*

## 1:00 — Why the system thinks so (25 s)

Click lead #1 (FIRs `b9612100ef5bc0ff` ↔ `5f14011fad04a685`).

- Top line: *rarer than 1 in 2,00,000 unrelated cases look this alike.*
  Measured on two lakh random unrelated pairs — never a probability.
- Walk down the reasons: the time gap first, then each shared habit with how
  common it is ("only 4% of house burglaries…"), then the differences, then
  what wasn't recorded.
- Type an officer name, click **Needs investigation**, show it land in the
  decision history. **Print report**.

## 1:25 — Location-blind vs nearby first (15 s)

Open any case (**Find a case** → an example), flip **Nearby first** and read
the line under the switch: *it nearly doubles how often the true partner is
in the top 10 (26% → 46%) but finds a cross-state partner for 3 cases instead
of 17. The officer chooses; the default stays location-blind, because that
is the point of this system.*

## 1:40 — Possible series (35 s)

**Series** → open the chain-snatching series across **Kolhapur, Nashik and Thane**
(id `2ca66ef385`, or filter State = Maharashtra, Crime type = Chain
snatching).

- Map and timeline: six FIRs, three districts, one MO.
- Be straight about it: *this one is clean — all one offender. Across all
  series, about 1 in 5 FIR pairs inside a series are the same offender, so
  every link is shown separately with its own strength.*

(We picked this series because it is a clean example. Saying so is part of
the demo.)

## 2:15 — Crime-type checks (20 s)

**Checks** → first card (`MP-IND-02/0059/2021`): recorded as house burglary,
but warehouse premises, shutter entry, closed for the night.

> 714 FIRs flagged; 97% really were filed under the wrong type. A misfiled
> FIR is compared with the wrong cases, so its real links are invisible —
> and its unusual details create false matches. The inbox pushes those leads
> to the bottom.

## 2:35 — हिंदी, and honesty (25 s)

- Click **हिंदी**: the whole interface switches; FIR text stays as recorded.
- **How it works**: the numbers, including the weak ones (cross-state, cross
  crime type). Safeguards: no caste/religion/community field anywhere,
  personal details never enter matching, location never scores a match,
  a human decides.

## If asked

- **"How accurate is it?"** Accuracy is the wrong measure — a true link is
  1 in ~26,000 pairs, so "never linked" is 99.996% accurate. For 26% of
  cases the true partner is in the top 10 of ~15,000 FIRs of that type
  (46% with "nearby first").
- **"Is this real data?"** No. Synthetic, generated to resemble FIRs from
  four states. The method's numbers must be re-measured on real records;
  the timing gain in particular comes from how we generated the data.
- **"Cost?"** Runs on the AWS free tier: one Lambda, two DynamoDB tables.
- **"Login?"** Officer names are typed in this demo and marked unverified in
  the audit log. Production adds sign-in (Cognito via API Gateway).
