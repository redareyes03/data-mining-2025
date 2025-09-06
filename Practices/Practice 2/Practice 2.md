## Practica 2

**Instrucciones:** Run descriptive statistics to data, identify the entities and relations and draw its diagram. Also group data by entities and obtain statistics from grouped data.

Nota: Respecto a la practica pasada se modifico un poco la forma en la que se trabajara
con los datasets, pues solo se habia incluido un dataset de 4521 filas, el cual corresponde
al dataset de testeo, se agregó la parte de entrenamiento. Estos se necuentran en
```/datasets```

### Entidades

1. Clients
    - Age (int)
    - Job (string - categorical)
    - Marital (string - categorical)
    - Education (stirng - categorical)
    - Default (boolean)
    - Balance (int)
    - Housing (boolean)
    - Loan (binary)
2. Contacts 
   - Contact (string - categorical)
   - Day (int)
   - Month (int)
   - Duration (seconds) (int)
   - Campaign (int)
   - Pdays (int)
   - Previous (int)
   - Poutcome (string - categorical)

### Clientes
#### Tasa de conversion por edad
![img_9.png](img_9.png)

#### Tasa de conversion por trabajo
![img.png](img.png)

#### Tasa de conversion por estado civil
![img_1.png](img_1.png)

#### Tasa conversion por nivel educativo
![img_2.png](img_2.png)

#### Tasa de conversion por incumplimientos de pago (default)
![img_3.png](img_3.png)

#### Tasa de conversion por posesion de prestamos hiptecarios
![img_4.png](img_4.png)

#### Tasa de conversion por posesion de creditos
![img_5.png](img_5.png)

### Contact
#### Tasa de conversion por semana
![img_6.png](img_6.png)
#### Tasa de conversion por tipo de contacto
![img_7.png](img_7.png)
#### Tasas de conversion por mes
![img_8.png](img_8.png)