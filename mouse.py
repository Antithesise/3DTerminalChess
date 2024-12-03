from textual import events
from textual.app import App
from textual.widget import Widget

pos = ""

class text(Widget):
    def render(self):
        return pos

class MouseApp(App):
    def compose(self):
        yield text()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        global pos
        pos = str(event)

if __name__ == "__main__":
    app = MouseApp()
    app.run()