# AegisOps Agent UI Redesign

## Goal

Rework the existing static chat frontend to match the provided warm orange AegisOps Agent reference screen while preserving the current FastAPI endpoints and client-side chat behavior.

## Scope

- Rename visible project branding to `AegisOps Agent`.
- Update the default backend application name to `AegisOps Agent`.
- Keep the current single-page static frontend under `static/`.
- Preserve existing chat, streaming chat, file upload, chat history, and AI Ops trigger behavior.
- Restyle the empty/home state to match the screenshot: left sidebar, warm ivory background, orange brand treatment, central assistant illustration, greeting copy, quick prompt chips, and large rounded input panel.
- Keep message rendering and markdown support intact once a conversation starts.

## Frontend Design

The page remains a two-column app shell. The sidebar becomes a 320 px warm panel with a shield logo, `AegisOps Agent` wordmark, an orange new-chat button, recent conversation cards, a small operations promo block, and secondary navigation rows. The main area uses a soft ivory background with the AI Ops pill in the top-right corner.

The empty chat state shows a centered hero panel with a lightweight CSS/SVG operations assistant illustration, the headline `你好！我是 AegisOps Agent`, and support text for operations questions, data analysis, report generation, and scripts. The input panel sits near the lower center with a large prompt area, quick action chips, mode selector, attachment control, and orange send button.

## Behavior

When there are no messages, the app displays the redesigned welcome state and centered input. When messages exist or a history entry is opened, the welcome hero hides and the message list behaves like the existing chat view. Existing JavaScript methods continue to own sending, streaming, upload, AI Ops, and history persistence.

## Testing

Add lightweight regression checks for static branding and required home-state elements, then run them before and after implementation. Verify the page visually through a local FastAPI server and browser screenshot when possible.

## Constraints

This workspace is not a Git repository, so the design document cannot be committed locally.
