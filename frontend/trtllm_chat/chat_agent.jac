import from httpx { post };
import from json { loads };

node Message {
    has role: str;
    has content: str;
}

node ConversationRoot {}

def get_llm_response(history: list[dict]) -> str {
    data = {"messages": history, "max_tokens": 1024, "stream": false};
    response = post("http://localhost:8082/v1/chat/completions", json=data);
    if response.status_code == 200 {
        js = loads(response.text);
        return js["choices"][0]["message"]["content"];
    } else {
        return "Error generating response";
    }
}

walker CollectHistory {
    has history: list[dict] = [];

    can collect with Message entry {
        self.history.append({"role": here.role, "content": here.content});
        visit [-->];
    }
}

walker:pub ChatAgent {
    has user_content: str;

    can process_chat with ConversationRoot entry {
        // Find the last message in the chain
        last = here;
        while last[-->] {
            last = last[-->][0];  // Assume single forward connection
        }

        // Add user message node
        user_msg = last ++> Message(role="user", content=self.user_content);

        // Collect full history using a collector walker
        collector = spawn CollectHistory();
        spawn here collector;
        history = collector.history;

        // Get LLM response from your TRT-LLM backend
        assistant_content = get_llm_response(history);

        // Add assistant message node
        user_msg ++> Message(role="assistant", content=assistant_content);

        // Report the response (becomes API response)
        report {"content": assistant_content};
    }
}

// Example usage: Run with `jac start chat_agent.jac` to expose /walker/ChatAgent as POST endpoint
// Body: {"user_content": "Hello"}
// In your Reflex app, you could swap the httpx call to hit the Jac endpoint instead for full integration.
