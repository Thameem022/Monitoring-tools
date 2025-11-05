import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app
import types
import app


def test_respond_function_exists():
    """Check that the app has a respond() function."""
    assert hasattr(app, "respond")
    assert callable(app.respond)


def test_respond_returns_generator():
    """respond() should return a generator when called with minimal args."""

    gen = app.respond(
        message="I'm buying a bottle of water.",
        history=[],
        hf_token_ui="", 
        system_message="You are Sustainable.ai.",
        car_km=10,
        bus_km=0,
        train_km=0,
        air_km=0,
        meat_meals=3,
        vegetarian_meals=2,
        vegan_meals=1,
    )

    import types
    assert isinstance(gen, types.GeneratorType)
