# __package__ = ""
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from .simple_tag.simple_tag import env, parallel_env, raw_env  # noqa: F401
