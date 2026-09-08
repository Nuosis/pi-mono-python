"""Provider stream boundary: a no-argument tool may emit no JSON deltas."""
from types import SimpleNamespace

import pytest

from pi_ai.models import get_model
from pi_ai.providers import anthropic
from pi_ai.types import Context, SimpleStreamOptions, Tool, UserMessage


@pytest.mark.parametrize('initial,deltas,expected,error',[
    ({},[],{},False),
    ({'limit':3},[],{'limit':3},False),
    ({},['{"limit":','4}'],{'limit':4},False),
    ({},['definitely invalid'],{},True),
])
async def test_stream_preserves_initial_tool_input(monkeypatch,initial,deltas,expected,error):
    class Stream:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def __aiter__(self):return events()
    client=SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs:Stream()))
    monkeypatch.setattr(anthropic,'_build_client',lambda *args,**kwargs:(client,False))
    def event(name,**values):return type(name,(),{})( ) if not values else type(name,(),values)()
    async def events(*args,**kwargs):
        yield event('RawContentBlockStartEvent',index=0,content_block=SimpleNamespace(type='tool_use',id='call',name='inventory',input=initial))
        for delta in deltas:
            yield event('RawContentBlockDeltaEvent',index=0,delta=SimpleNamespace(type='input_json_delta',partial_json=delta))
        yield event('RawContentBlockStopEvent',index=0)
    context=Context(messages=[UserMessage(role='user',content='Inspect inventory',timestamp=1)],
                    tools=[Tool(name='inventory',description='Read inventory',parameters={'type':'object','properties':{}})])
    output=[x async for x in anthropic.stream_simple(get_model('anthropic','claude-sonnet-4-6'),context,SimpleStreamOptions(api_key='test'))]
    assert any(x.type=="toolcall_end" for x in output), [x.model_dump() for x in output]
    call=next(x.tool_call for x in output if x.type=='toolcall_end')
    assert call.arguments==expected
    assert bool(call.arguments_parse_error)==error
