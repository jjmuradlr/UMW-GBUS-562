# Week 2 Writeup: Prompting Techniques for Wildframe Coffee Roasters

## What I Built
I built prompt-lab.html, a single-page Prompt Lab that calls the Anthropic API using the claude-sonnet-5 model. It has a System box, a message box, and an Export button that downloads the conversation as a Markdown file with the model name on the first line and each message under a System, You, or Claude heading. The API key is pasted into the app each session and is never saved to any file. I used the lab to run the four Wildframe tasks, each in a new conversation, and structured every prompt with a role in the System box followed by Context, Task, Format, and Constraints in the message. All four runs met their required format on the first response, and the exports are saved in runs/task-1.md through runs/task-4.md.

## Why This Approach
Task 1 - Technique: zero-shot
Generating perk ideas is open-ended brainstorming, and the Brand Kit has no past perk examples I could use. Adding unrelated examples like a bean card or customer reply would not teach the model anything about perks and could pull its ideas toward whatever the example was about. A clear role, the customer profile, and tight format rules (exactly 6 numbered ideas of 25 words or fewer) were enough for the model to work from.

Task 2 - Technique: one-shot
The bean cards are mostly a format-copying task: the origin line in capitals, then the same four labels in the same order. The Brand Kit has exactly one house bean card, and one clean example shows the structure and the tone of the Real talk line better than any description could. More examples were not available and were not needed, because the facts for each coffee were supplied in the prompt.

Task 3 - Technique: few-shot
Brand voice is hard to describe and easier to show. "Warm, plainspoken, and a little irreverent" can be read many ways, so I gave the model all four past customer replies from the Brand Kit. Together they show how short Wildframe's replies are, how they handle both complaints and praise, and how they often end on a light, dry joke. The three new comments cover the same range (a complaint, a product question, and a compliment), so several varied examples gave the model a pattern for each kind.

Task 4 - Technique: chain-of-thought
Choosing the pop-up location requires multi-step math (buyers, weekly revenue, four-week revenue, four-week rent, and revenue minus rent) plus a check against the $5,000 rent budget. Asking the model to reason step by step makes each calculation visible, so mistakes are easier to catch and the budget rule is applied explicitly instead of skipped. I included all of the location data in the prompt, along with a short note about Wildframe's business, so the final choice could weigh both the numbers and fit with whole-bean bag and subscription sales.

## Example Run
Model: claude-sonnet-5

### System
You are Wildframe Coffee Roasters' social media and customer care voice. You reply to customer comments the way Wildframe always has: warm, plainspoken, and a little irreverent, with no corporate jargon.

### You
CONTEXT: Wildframe's customers are environmentally conscious urban professionals. Here are past customer comments and the replies Wildframe posted. Match their voice, length, and style.

Comment: Is your packaging actually recyclable, or is that just marketing?
Reply: Fair question. The bag is compostable, the little valve isn't, so pop the valve off first. We're working on that last bit.

Comment: Your Kenya roast tasted burnt to me.
Reply: That's not the cup we were going for. Tell us how you brewed it and we'll help you dial it in, or swap you to something mellower.

Comment: Best coffee I've had in years!!
Reply: This made our whole roasting crew's morning. We'll keep doing the thing.

Comment: Can I pause my subscription while I travel?
Reply: Yep. Account, then Subscription, then Pause. Your beans will wait for you, mostly patiently.

TASK: Write Wildframe's reply to each of these three new comments:
1. My order was supposed to arrive Monday. It's Thursday.
2. Do you make any decaf that doesn't taste like cardboard?
3. The Huila is my new Sunday morning ritual. Thank you.

FORMAT: Three replies numbered 1 to 3, matching the comment numbers. Each reply is no more than 40 words. Output only the three numbered replies.

CONSTRAINTS:
- No emoji.
- No corporate jargon or scripted apology language ("We apologize for any inconvenience").
- Do not invent specific product names, prices, refund amounts, or policies that were not given. If a next step is needed, invite the customer to message or email the team.
- Count the words in each reply before answering and shorten any that exceed 40.

### Claude
1. That's three days too many. Send us your order number and we'll track it down, no runaround, promise.

2. We do, and it actually tastes like coffee. DM us and we'll point you to the right bag.

3. Okay, this is the best kind of email to read on a Monday. Enjoy every Sunday, seriously.

## Reflection
The technique choice that made the biggest difference was few-shot for the customer replies. The four past replies taught the model Wildframe's rhythm better than any adjective could: short sentences, a direct answer, and a light closing line. "That's three days too many" sounds like the same brand that wrote "Your beans will wait for you, mostly patiently." The examples also exposed a limit, though. They carried the voice but not the facts. Reply 2 says Wildframe makes a good decaf, which the prompt never stated, and Reply 3 calls a comment an "email." Few-shot copies style, and it still needs explicit constraints for accuracy.

One-shot worked well for the bean cards because the format was the hard part, and a single card showed it exactly. The Real talk lines did run longer than the example's clipped two-sentence joke. Chain-of-thought mattered most for trust rather than for the answer itself. Riverside was probably going to win either way, but the step-by-step math showed Union Station ruled out on rent alone and made every number checkable.

The choice I'm least sure of is zero-shot for the perks. The ideas were usable but generic: early access, a tote, tree planting. These are perks many brands already offer. Next time I would try chain-of-thought, asking the model to first list what a small roaster can actually afford and what eco-conscious subscribers care about, then generate perks from that reasoning.
