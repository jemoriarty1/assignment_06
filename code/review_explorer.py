# 1. WHICH REVIEWS
reviews_file = os.environ.get("REVIEWS_FILE", REVIEWS_FILE)

# 2. HEADING
st.title("Salt City Coffee — What Customers Are Saying")
st.markdown(
    "Every customer review, broken into the things people talk about. "
    "Filter by location or subject, or ask a question at the bottom of the page."
)

# 3. THE DATA
mentions = analyze_reviews(reviews_file)

# 4. TWO FILTERS, SIDE BY SIDE
left, right = st.columns(2)
with left:
    location = st.selectbox(
        "Location",
        ["All"] + sorted(mentions["location"].unique()),
        key="location",
    )
with right:
    subject = st.selectbox(
        "Subject",
        ["All"] + sorted(mentions["subject"].unique()),
        key="subject",
    )

selected = mentions
if location != "All":
    selected = selected[selected["location"] == location]
if subject != "All":
    selected = selected[selected["subject"] == subject]

# 5. FOUR METRICS, SIDE BY SIDE
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reviews", selected["review_id"].nunique())
col2.metric("Mentions", len(selected))
col3.metric("Positive", int((selected["mention_sentiment"] == "positive").sum()))
col4.metric("Negative", int((selected["mention_sentiment"] == "negative").sum()))

# 6. THREE TABLES IN THREE TABS
summary = summarize_items(selected)
praised, complained, said = st.tabs(
    ["Most praised", "Most complained about", "What people said"]
)
praised.dataframe(
    summary[summary["net"] > 0].sort_values("net", ascending=False).head(5),
    hide_index=True,
)
complained.dataframe(
    summary[summary["net"] < 0].sort_values("net").head(5),
    hide_index=True,
)
said.dataframe(
    selected[["date", "location", "item", "subject", "mention_sentiment",
              "assessments", "sentence_text"]],
    hide_index=True,
)

# 7. THE QUESTION BOX
st.subheader("Ask the reviews")
question = st.text_input(
    "Your question",
    key="question",
    placeholder="What is the most popular product?",
)
ask = st.button("Ask", key="ask")

# 8. WHEN ASK IS CLICKED
if ask and question.strip():
    scope, evidence = answer_question(question, mentions, load_aliases())
    if evidence.empty:
        st.warning(
            f"No reviews mention {describe_scope(scope)}. Try asking about a "
            "menu item, a staff member, or part of the shop."
        )
    else:
        messages = build_messages(question, evidence)
        with st.spinner("Asking the model..."):
            answer = ask_llm(messages)
        st.markdown(answer)
        st.caption(f"Based on {len(evidence)} mentions — {describe_scope(scope)}:")
        st.dataframe(
            evidence[["date", "location", "item", "mention_sentiment",
                      "sentence_text"]],
            hide_index=True,
        )
