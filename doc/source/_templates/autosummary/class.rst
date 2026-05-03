{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :show-inheritance:

{% block methods %}
{% set own_public = methods | reject('in', inherited_members) | list %}
{% set own_dunders = all_methods | select('in', ['__call__', '__len__', '__getitem__', '__mul__']) | reject('in', inherited_members) | list %}
{% if own_public or own_dunders %}
.. rubric:: Methods

.. autosummary::
   :toctree:
   :nosignatures:

   {% for item in own_public %}
   {%- if item != '__init__' %}
   {{ name }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% for item in own_dunders %}
   {{ name }}.{{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% set own_attrs = attributes | reject('in', inherited_members) | list %}
{% if own_attrs %}
.. rubric:: Attributes

.. autosummary::
   :toctree:

   {% for item in own_attrs %}
   {%- if not item.startswith('_') %}
   {{ name }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
{% endif %}
{% endblock %}