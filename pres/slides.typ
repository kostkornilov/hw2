#import "@preview/touying:0.8.0": *
#import themes.simple: title-slide

#set text(font: "New Computer Modern")

#show: touying-slides.with(
  title: [Sparse Attention as Compact Kernel Regression],
  subtitle: [Объяснение статьи],
  author: [Константин Корнилов],
  date: 10-10-2026,
)

#title-slide[
  #text(size: 1.8em, weight: "bold")[Sparse Attention as Compact Kernel Regression]
  
  #text(size: 1.2em)[Объяснение статьи]
  
  Константин Корнилов
  
  10 октября 2026
]

#slide[
  = Proposed method

  #align(center)[
    #image(
      "figures/kernels.png",
      width: 70%,
    )
  ]

  Attention formula


  $ "Attention"(Q, K, V) = 
    #text(fill: rgb("d9383a"))[$"softmax"$] 
    
    #box(
      stroke: 1pt + rgb("1e62d0"), 
      inset: (x: 3pt, y: 5pt), 
      baseline: 0%
    )[
      $ (Q K^T) / sqrt(d_k) $
    ] 
    V 
  $
]