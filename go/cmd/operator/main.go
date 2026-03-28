// Package main is the entrypoint for the Podstack operator.
// It sets up the controller manager and registers all reconcilers.
package main

import (
	"flag"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	v1 "github.com/podstack/serverless/api/v1"
	"github.com/podstack/serverless/internal/controller"
	"github.com/podstack/serverless/internal/snapshot"
	"github.com/podstack/serverless/pkg/nfs"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1.AddToScheme(scheme))
}

func main() {
	var (
		metricsAddr          string
		healthProbeAddr      string
		enableLeaderElection bool
		nfsBasePath          string
		snapshotBasePath     string
	)

	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&healthProbeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false, "Enable leader election for controller manager.")
	flag.StringVar(&nfsBasePath, "nfs-base-path", "/models", "Base path for NFS model cache.")
	flag.StringVar(&snapshotBasePath, "snapshot-base-path", "/snapshots", "Base path for snapshot blobstore.")
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		HealthProbeBindAddress: healthProbeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "podstack-operator-leader",
	})
	if err != nil {
		setupLog.Error(err, "unable to create manager")
		os.Exit(1)
	}

	log := ctrl.Log.WithName("operator")

	// Initialize shared components
	modelCache := nfs.NewModelCache(nfsBasePath, log.WithName("nfs"))
	snapshotMgr := snapshot.NewManager(mgr.GetClient(), snapshotBasePath, log.WithName("snapshot"))

	// Register controllers
	if err := (&controller.ModelDeploymentReconciler{
		Client:          mgr.GetClient(),
		Scheme:          mgr.GetScheme(),
		SnapshotManager: snapshotMgr,
		ModelCache:      modelCache,
		Log:             log.WithName("modeldeployment"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "ModelDeployment")
		os.Exit(1)
	}

	if err := (&controller.SnapshotReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
		Log:    log.WithName("snapshot"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Snapshot")
		os.Exit(1)
	}

	if err := (&controller.LoRAAdapterReconciler{
		Client:     mgr.GetClient(),
		Scheme:     mgr.GetScheme(),
		ModelCache: modelCache,
		Log:        log.WithName("loraadapter"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "LoRAAdapter")
		os.Exit(1)
	}

	if err := (&controller.TenantReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
		Log:    log.WithName("tenant"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Tenant")
		os.Exit(1)
	}

	// Health checks
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting Podstack operator",
		"nfsBasePath", nfsBasePath,
		"snapshotBasePath", snapshotBasePath,
	)

	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
