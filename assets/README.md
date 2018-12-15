# Sponza Atrium

To extract:

```
unzip sponza.zip -d Sponza
sed -i 's/\\/\//g' Sponza/sponza.mtl
```

To render:

```
path_tracer --scene assets/sponza.json
```
